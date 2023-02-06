# Use both diffusion model (seperate models) as prior and posterior

import argparse
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pytorch_fid_wrapper as pfw
import shutil
import datetime as dt
import re
from src.diffusion_net import _netG_cifar10, _netE, _netQ, _netQ_uncond, _netQ_U
from src.MCMC import sample_langevin_post_z_with_diffgrad, sample_langevin_post_z_with_gaussian, gen_samples_with_diffusion_prior, calculate_fid_with_diffusion_prior


torch.multiprocessing.set_sharing_strategy('file_system')

#################### training #####################################

def main(args):

    # initialize random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # make log directory  
    timestamp = str(dt.datetime.now())[:19]
    timestamp = re.sub(r'[\:-]','', timestamp) # replace unwanted chars 
    timestamp = re.sub(r'[\s]','_', timestamp) # with regex and re.sub
    
    img_dir = os.path.join(args.log_path, timestamp, 'imgs')
    ckpt_dir = os.path.join(args.log_path, timestamp, 'ckpt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.log_path, timestamp, osp.basename(__file__)))

    # load dataset and calculate statistics
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_iter = iter(trainloader)

    start_time = time.time()
    print("Begin calculating real image statistics")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    mset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    mloader = data.DataLoader(mset, batch_size=500, shuffle=False, num_workers=1, drop_last=False)
    
    # pre-calculating statistics for fid calculation
    fid_data_true = []
    for x, _ in testloader:
        fid_data_true.append(x)
        if len(fid_data_true) >= args.n_fid_samples:
            break
    fid_data_true = torch.cat(fid_data_true, dim=0)
    fid_data_true = (fid_data_true + 1.0) / 2.0
    real_m, real_s = pfw.get_stats(fid_data_true, device="cuda:0")
    print("Finish calculating real image statistics {:.3f}".format(time.time() - start_time), fid_data_true.shape, fid_data_true.min(), fid_data_true.max())
    fid_data_true, testset, testloader = None, None, None

    # define models
    G = _netG_cifar10(nz=args.nz, ngf=args.ngf, nc=args.nc)
    Q = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A')
    Q_dummy = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A')
    for param, target_param in zip(Q.parameters(), Q_dummy.parameters()):
        target_param.data.copy_(param.data)

    G.cuda()
    Q.cuda()
    Q_dummy.cuda()

    # G_optimizer = optim.Adam(G.parameters(), lr=1e-5, betas=(0.5, 0.999))
    # Q_optimizer = optim.Adam(Q.parameters(), lr=1e-5, betas=(0.5, 0.999))

    # G_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    # Q_optimizer = optim.Adam(Q.parameters(), lr=args.q_lr, betas=(0.5, 0.999))

    G_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    Q_optimizer = optim.AdamW(Q.parameters(), weight_decay=0.01, lr=args.q_lr, betas=(0.5, 0.999))

    start_iter = 0
    fid_best = 10000
    mse_best = 10000
    if args.resume_path is not None:
        print('load from ', args.resume_path)
        state_dict = torch.load(args.resume_path)
        G.load_state_dict(state_dict['G_state_dict'])
        Q.load_state_dict(state_dict['Q_state_dict'])
        G_optimizer.load_state_dict(state_dict['G_optimizer'])
        Q_optimizer.load_state_dict(state_dict['Q_optimizer'])
        start_iter = state_dict['iter'] + 1
    
    g_lr = args.g_lr
    q_lr = args.q_lr
    rho = 0.005
    p_mask = args.p_mask

    # begin training
    for iteration in range(start_iter, args.iterations + 1):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            x, _ = next(train_iter)
        x = x.cuda()

        Q.eval()
        G.eval()
        # infer z from given x
        with torch.no_grad():
            z0 = Q_dummy(x)
        zk_pos = z0.detach().clone()
        zk_pos.requires_grad = True
        zk_pos = sample_langevin_post_z_with_gaussian(z=zk_pos, x=x, netG=G, netE=Q, g_l_steps=args.g_l_steps, g_llhd_sigma=args.g_llhd_sigma, g_l_with_noise=args.g_l_with_noise, \
            g_l_step_size=args.g_l_step_size, verbose = (iteration % (args.print_iter * 10) == 0))
        
        # update Q 
        Q_optimizer.zero_grad()
        Q.train()

        z_mask_prob = torch.rand((len(zk_pos),)).to(zk_pos.device)
        z_mask = torch.ones(len(zk_pos),).to(zk_pos.device)
        z_mask[z_mask_prob < p_mask] = 0.0
        z_mask = z_mask.unsqueeze(-1)
        a_mask = torch.ones(len(zk_pos),).to(zk_pos.device)

        Q_loss_t = Q.calculate_loss(x=x, z=zk_pos, mask=z_mask).mean()
        Q_loss_0 = Q.calculate_loss(x=x, z=z0, mask=a_mask).mean() * 1e-2
        Q_loss = Q_loss_t - Q_loss_0
        Q_loss.backward()
        if args.q_is_grad_clamp:
            torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=args.q_max_norm)
        Q_optimizer.step()
        
        # update G
        G_optimizer.zero_grad()
        G.train()

        # x0 = G(z0)
        x_hat = G(zk_pos)
        g_loss_t = torch.sum((x_hat - x) ** 2, dim=[1,2,3]).mean()
        # g_loss_0 = torch.sum((x0 - x) ** 2, dim=[1,2,3]).mean() * 1e-3
        g_loss = g_loss_t # - g_loss_0
        g_loss.backward()
        if args.g_is_grad_clamp:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.g_max_norm)
        G_optimizer.step()

        Q.eval()
        G.eval()
        # learning rate schedule
        if (iteration + 1) % 1000 == 0:
            g_lr = max(g_lr * 0.99, 1e-5)
            q_lr = max(q_lr * 0.99, 1e-5)
            for G_param_group in G_optimizer.param_groups:
                G_param_group['lr'] = g_lr
            for Q_param_group in Q_optimizer.param_groups:
                Q_param_group['lr'] = q_lr

        if (iteration + 1) % 10 == 0:
            # Update the frozen target models
            for param, target_param in zip(Q.parameters(), Q_dummy.parameters()):
                target_param.data.copy_(rho * param.data + (1 - rho) * target_param.data)


        if iteration % args.print_iter == 0:
            # print("Iter {} time {:.2f} g_loss {:.6f} q_loss {:.3f} g_lr {:.8f} q_lr {:.8f}".format(
            #     iteration, time.time() - start_time, g_loss.item(), Q_loss.item(), g_lr, q_lr))
            print("Iter {} time {:.2f} g_loss {:.6f}|{:.6f} q_loss {:.3f}|{:.6f} g_lr {:.8f} q_lr {:.8f}".format(
                iteration, time.time() - start_time, g_loss.item(), g_loss_t.item(), 
                                                     Q_loss.item(), Q_loss_t.item(), g_lr, q_lr))
            print(zk_pos.max(), zk_pos.min())
        if iteration % args.plot_iter == 0:
            # reconstruction
            with torch.no_grad():
                x_hat_q = G(z0)
                save_images = x[:64].detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_obs.png'.format(img_dir, iteration), normalize=True, nrow=8)
                save_images = x_hat[:64].detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_post.png'.format(img_dir, iteration), normalize=True, nrow=8)
                save_images = x_hat_q[:64].detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_post_Q.png'.format(img_dir, iteration), normalize=True, nrow=8)
            # samples
            samples, _ = gen_samples_with_diffusion_prior(b=64, device=z0.device, netE=Q, netG=G) 
            save_images = samples[:64].detach().cpu()
            torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_prior.png'.format(img_dir, iteration), normalize=True, nrow=8)
        
        if iteration > 0 and iteration % args.ckpt_iter == 0:
            print('Saving checkpoint')
            save_dict = {
                'G_state_dict': G.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'Q_state_dict': Q.state_dict(),
                'Q_optimizer': Q_optimizer.state_dict(),
                'Q_dummy_state_dict': Q_dummy.state_dict(),
                'iter': iteration
            }
            torch.save(save_dict, os.path.join(ckpt_dir, '{}.pth.tar'.format(iteration)))
        
        if iteration % args.fid_iter == 0:
            fid_s_time = time.time()
            out_fid = calculate_fid_with_diffusion_prior(n_samples=args.n_fid_samples, device=z0.device, netE=Q, netG=G, real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, iteration))
            if out_fid < fid_best:
                fid_best = out_fid
                print('Saving best checkpoint')
                save_dict = {
                    'G_state_dict': G.state_dict(),
                    'G_optimizer': G_optimizer.state_dict(),
                    'Q_state_dict': Q.state_dict(),
                    'Q_optimizer': Q_optimizer.state_dict(),
                    'Q_dummy_state_dict': Q_dummy.state_dict(),
                    'iter': iteration
                }
                torch.save(save_dict, os.path.join(ckpt_dir, 'best.pth.tar'))
            print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(time.time() - fid_s_time, out_fid, fid_best))

            mse_lss = 0.0
            mse_s_time = time.time()

            i = 0
            for x, _ in mloader:
                x = x.cuda()
                with torch.no_grad():
                    z0 = Q_dummy(x)
                zk_pos = z0.detach().clone()
                zk_pos.requires_grad = True
                zk_pos = sample_langevin_post_z_with_gaussian(
                            z=zk_pos, x=x, netG=G, netE=Q, g_l_steps=10, g_llhd_sigma=args.g_llhd_sigma, g_l_with_noise=False,
                            g_l_step_size=args.g_l_step_size, verbose=False
                        )

                with torch.no_grad():
                    x_hat = G(zk_pos)
                    g_loss = torch.mean((x_hat - x) ** 2, dim=[1,2,3]).sum()
                mse_lss += g_loss.item()

            mse_lss /= len(mset)
            if mse_lss < mse_best:
                mse_best = mse_lss
            print("Finish calculating mse time {:.3f} mse {:.3f} / {:.3f}".format(time.time() - mse_s_time, mse_lss, mse_best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_path', type=str, default='../logs/cifar', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    parser.add_argument('--resume_path', type=str, default=None, help='pretrained ckpt path for resuming training')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--nc', type=int, default=3, help='image channel')
    parser.add_argument('--n_fid_samples', type=int, default=50000, help='number of samples for calculating fid during training')
    
    # network structure related parameters
    parser.add_argument('--nz', type=int, default=100, help='z vector length')
    parser.add_argument('--ngf', type=int, default=128, help='base channel numbers in G')
    parser.add_argument('--nif', type=int, default=64, help='base channel numbers in Q encoder')
    parser.add_argument('--nxemb', type=int, default=1024, help='x embedding dimension in Q')
    parser.add_argument('--ntemb', type=int, default=128, help='t embedding dimension in Q')

    # latent diffusion related parameters
    parser.add_argument('--n_interval_posterior', type=int, default=100, help='number of diffusion steps used here')
    parser.add_argument('--n_interval_prior', type=int, default=100, help='number of diffusion steps used here')
    parser.add_argument('--logsnr_min', type=float, default=-5.1, help='minimum value of logsnr')
    parser.add_argument('--logsnr_max', type=float, default=9.8, help='maximum value of logsnr')
    parser.add_argument('--diffusion_residual', type=bool, default=True, help='whether treat prediction as residual in latent diffusion model')
    parser.add_argument('--var_type', type=str, default='large', help='variance type of latent diffusion')
    parser.add_argument('--Q_with_noise', type=bool, default=True, help='whether include noise during inference')
    parser.add_argument('--p_mask', type=float, default=0.2, help='probability of prior model')
    parser.add_argument('--cond_w', type=float, default=0.0, help='weight of conditional guidance')
    
    # MCMC related parameters
    parser.add_argument('--g_l_steps', type=int, default=30, help='number of langevin steps for posterior inference')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of posterior langevin')
    parser.add_argument('--g_l_with_noise', default=False, type=bool, help='noise term of posterior langevin')
    parser.add_argument('--g_llhd_sigma', type=float, default=0.1, help='sigma for G loss')
    parser.add_argument('--e_l_steps', type=int, default=60, help='number of langevin steps for prior sampling')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of prior langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of prior langevin')

    # optimizing parameters
    parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--e_lr', type=float, default=5e-5, help='learning rate for latent ebm')
    parser.add_argument('--q_lr', type=float, default=2e-4, help='learning rate for inference model Q')
    parser.add_argument('--q_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--q_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--iterations', type=int, default=1000000, help='total number of training iterations')
    parser.add_argument('--print_iter', type=int, default=100, help='number of iterations between each print')
    parser.add_argument('--plot_iter', type=int, default=1000, help='number of iterations between each plot')
    parser.add_argument('--ckpt_iter', type=int, default=50000, help='number of iterations between each ckpt saving')
    parser.add_argument('--fid_iter', type=int, default=500, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)
