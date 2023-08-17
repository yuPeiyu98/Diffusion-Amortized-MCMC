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
from data.dataset import MNIST
from src.diffusion_net import _netG_cifar10, _netG_svhn, _netG_celeba64, _netG_celebaHQ, _netE, _netQ, _netQ_uncond, _netQ_U
from src.diffusion_net import Generator, _netG_cifar10_deep
from src.MCMC import sample_langevin_post_z_with_prior, sample_langevin_prior_z, sample_langevin_post_z_with_gaussian
from src.MCMC import gen_samples_with_diffusion_prior, calculate_fid_with_diffusion_prior, calculate_fid


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
    
    img_dir = os.path.join(args.log_path, args.dataset, timestamp, 'imgs')
    ckpt_dir = os.path.join(args.log_path, args.dataset, timestamp, 'ckpt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.log_path, args.dataset, timestamp, osp.basename(__file__)))

    # load dataset and calculate statistics
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.dataset == 'cifar10':
        args.nz = 128
        args.ngf = 128
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
        mset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == 'svhn':
        args.nz = 100
        args.ngf = 64
        trainset = torchvision.datasets.SVHN(root=args.data_path, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=args.data_path, split='train', download=True, transform=transform_test) 
        mset = torchvision.datasets.SVHN(root=args.data_path, split='test', download=True, transform=transform_test)
    elif args.dataset == 'celeba64':
        args.nz = 100
        args.ngf = 128

        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'celeba64_train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'celeba64_train'), transform=transform_test) 
        mset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'celeba64_test'), transform=transform_test)
    elif args.dataset == 'celebaHQ':
        args.nz = 128
        args.ngf = 128

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'val'), transform=transform_test) 
        mset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'test'), transform=transform_test)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    mloader = data.DataLoader(mset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    train_iter = iter(trainloader)
    
    # pre-calculating statistics for fid calculation
    start_time = time.time()
    print("Begin calculating real image statistics")
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
    if args.dataset == 'cifar10':
        # G = _netG_cifar10(nz=args.nz, ngf=args.ngf, nc=args.nc)
        G = _netG_cifar10_deep(nz=args.nz, ngf=args.ngf, nc=args.nc)
        # G = Generator(z_dim=args.nz)
    elif args.dataset == 'svhn':
        G = _netG_svhn(nz=args.nz, ngf=args.ngf, nc=args.nc)
    elif args.dataset == 'celeba64':
        G = _netG_celeba64(nz=args.nz, ngf=args.ngf, nc=args.nc)
    elif args.dataset == 'celebaHQ':
        G = _netG_celebaHQ(nz=args.nz, ngf=args.ngf, nc=args.nc)
    NF = 4
    Q = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nf=NF, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_dummy = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nf=NF, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_eval = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nf=NF, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    for param, target_param in zip(Q.parameters(), Q_dummy.parameters()):
        target_param.data.copy_(param.data)
    for param, target_param in zip(Q.parameters(), Q_eval.parameters()):
        target_param.data.copy_(param.data)

    E = _netE(nz=args.nz)

    G.cuda()
    Q.cuda()
    E.cuda()
    Q_dummy.cuda()
    Q_eval.cuda()

    G_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    Q_optimizer = optim.AdamW(Q.parameters(), weight_decay=1e-4, lr=args.q_lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=args.e_lr, betas=(0.5, 0.999))

    start_iter = 0
    fid_best = 10000
    fid_best_ = 10000
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
    e_lr = args.e_lr
    rho = 0.005
    p_mask = args.p_mask

    # begin training
    for iteration in range(start_iter, args.iterations + 1):
        try:
            x, idx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            x, idx = next(train_iter)
        x = x.cuda()
        # print(idx)

        z_mask_prob = torch.rand((len(x),), device=x.device)
        z_mask = torch.ones(len(x), device=x.device)
        z_mask[z_mask_prob < p_mask] = 0.0
        z_mask = z_mask.unsqueeze(-1)

        Q.eval()
        G.eval()
        E.eval()
        # infer z from given x
        with torch.no_grad():
            z0 = Q_dummy(x)
            zp = Q(x=None, b=x.size(0), device=x.device)
        zk_pos, zk_neg = z0.detach().clone(), z0.detach().clone()
        zk_pos.requires_grad = True
        zk_neg.requires_grad = True

        zk_pos = sample_langevin_post_z_with_prior(
            z=zk_pos, x=x, netG=G, netE=E, g_l_steps=args.g_l_steps, g_llhd_sigma=args.g_llhd_sigma, g_l_with_noise=args.g_l_with_noise,
            g_l_step_size=args.g_l_step_size, verbose = (iteration % (args.print_iter * 10) == 0))
        zk_neg = sample_langevin_prior_z(
            z=torch.cat([zk_neg, torch.randn_like(zk_neg, requires_grad=True)], dim=0), 
            netE=E, e_l_steps=args.e_l_steps, e_l_step_size=args.e_l_step_size, 
            e_l_with_noise=args.e_l_with_noise, verbose=False)
        # z_mask = torch.ones(len(x), device=x.device).unsqueeze(-1)
        
        for __ in range(6):
            # update Q 
            Q_optimizer.zero_grad()
            Q.train()
            # Q_loss_p = Q.calculate_loss(x=x, z=zk_pos, mask=z_mask).mean()
            # Q_loss_n = Q.calculate_loss(x=x, z=zk_neg, mask=1 - z_mask).mean()
            # Q_loss = Q_loss_p + Q_loss_n

            Q_loss = Q.calculate_loss(x=x, z=zk_pos, mask=z_mask).mean()
            Q_loss.backward()
            if args.q_is_grad_clamp:
                torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=args.q_max_norm)
            Q_optimizer.step()
        
        # update G
        G_optimizer.zero_grad()
        G.train()

        x_hat = G(zk_pos)
        g_loss = torch.sum((x_hat - x) ** 2, dim=[1,2,3]).mean()
        g_loss.backward()
        if args.g_is_grad_clamp:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.g_max_norm)
        G_optimizer.step()

        # update E
        E_optimizer.zero_grad()
        E.train()
        e_pos, e_neg = E(zk_pos), E(zk_neg)
        E_loss = e_pos.mean() - e_neg.mean() # + (e_pos ** 2).mean() + (e_neg ** 2).mean()
        E_loss.backward()
        if args.e_is_grad_clamp:
            torch.nn.utils.clip_grad_norm_(E.parameters(), max_norm=args.e_max_norm)
        E_optimizer.step()

        Q.eval()
        G.eval()
        E.eval()
        # learning rate schedule
        if (iteration + 1) % 1000 == 0:
            g_lr = max(g_lr * 0.99, 1e-5)
            q_lr = max(q_lr * 0.99, 1e-5)
            e_lr = max(e_lr * 0.99, 1e-5)
            for G_param_group in G_optimizer.param_groups:
                G_param_group['lr'] = g_lr
            for Q_param_group in Q_optimizer.param_groups:
                Q_param_group['lr'] = q_lr
            for E_param_group in E_optimizer.param_groups:
                E_param_group['lr'] = e_lr

        if (iteration + 1) % 10 == 0:
            # Update the frozen target models
            for param, target_param in zip(Q.parameters(), Q_dummy.parameters()):
                target_param.data.copy_(rho * param.data + (1 - rho) * target_param.data)

        # for param, target_param in zip(Q.parameters(), Q_eval.parameters()):
        #     target_param.data.copy_(rho * param.data + (1 - rho) * target_param.data)
        # if (iteration + 1) % 1000 == 0:
        #     # Update the frozen target models
        #     for param, target_param in zip(Q_eval.parameters(), Q_dummy.parameters()):
        #         target_param.data.copy_(param.data)

        if iteration % args.print_iter == 0:
            # print("Iter {} time {:.2f} g_loss {:.6f} q_loss {:.3f} g_lr {:.8f} q_lr {:.8f}".format(
            #     iteration, time.time() - start_time, g_loss.item(), Q_loss.item(), g_lr, q_lr))
            print("Iter {} time {:.2f} g_loss {:.6f} q_loss {:.3f} g_lr {:.8f} q_lr {:.8f}".format(
                iteration, time.time() - start_time, g_loss.item(), Q_loss.item(), g_lr, q_lr))
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
            samples, _ = gen_samples_with_diffusion_prior(b=64, device=z0.device, netQ=Q, netG=G) 
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
                'Q_eval_state_dict': Q_eval.state_dict(),
                'E_state_dict': E.state_dict(),
                'E_optimizer': E_optimizer.state_dict(),
                'iter': iteration
            }
            torch.save(save_dict, os.path.join(ckpt_dir, '{}.pth.tar'.format(iteration)))
        
        if iteration % args.fid_iter == 0:
            fid_s_time = time.time()
            out_fid = calculate_fid_with_diffusion_prior(
                n_samples=5000, # args.n_fid_samples, 
                device=z0.device, netQ=Q, netG=G, netE=E,
                real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, iteration), bs=args.batch_size)
            out_fid_ = calculate_fid(
                n_samples=5000, # args.n_fid_samples, 
                nz=args.nz, netG=G, netE=E,
                e_l_steps=args.e_l_steps, e_l_step_size=args.e_l_step_size, e_l_with_noise=args.e_l_with_noise,
                real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, "test"), bs=args.batch_size)
            if out_fid_ < fid_best_:
                fid_best_ = out_fid_
            if out_fid < fid_best:
                fid_best = out_fid
                print('Saving best checkpoint')
                save_dict = {
                    'G_state_dict': G.state_dict(),
                    'G_optimizer': G_optimizer.state_dict(),
                    'Q_state_dict': Q.state_dict(),
                    'Q_optimizer': Q_optimizer.state_dict(),
                    'Q_dummy_state_dict': Q_dummy.state_dict(),
                    'Q_eval_state_dict': Q_eval.state_dict(),
                    'E_state_dict': E.state_dict(),
                    'E_optimizer': E_optimizer.state_dict(),
                    'iter': iteration
                }
                torch.save(save_dict, os.path.join(ckpt_dir, 'best.pth.tar'))
            print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f} | ebm {:.3f} / {:.3f}".format(
            	time.time() - fid_s_time, out_fid, fid_best, out_fid_, fid_best_))

            mse_lss = 0.0
            mse_s_time = time.time()

            i = 0
            for x, _ in mloader:
                x = x.cuda()
                with torch.no_grad():
                    z0 = Q(x)
                zk_pos = z0.detach().clone()
                zk_pos.requires_grad = True
                zk_pos = sample_langevin_post_z_with_prior(
                            z=zk_pos, x=x, netG=G, netE=E, g_l_steps=10, # if out_fid > fid_best else 40, 
                            g_llhd_sigma=args.g_llhd_sigma, g_l_with_noise=False,
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
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--log_path', type=str, default='../logs/', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    parser.add_argument('--resume_path', type=str, default=None, help='pretrained ckpt path for resuming training')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--nc', type=int, default=3, help='image channel')
    parser.add_argument('--n_fid_samples', type=int, default=10000, help='number of samples for calculating fid during training')
    
    # network structure related parameters
    parser.add_argument('--nz', type=int, default=128, help='z vector length')
    parser.add_argument('--ngf', type=int, default=128, help='base channel numbers in G')
    parser.add_argument('--nif', type=int, default=64, help='base channel numbers in Q encoder')
    parser.add_argument('--nxemb', type=int, default=1024, help='x embedding dimension in Q')
    parser.add_argument('--ntemb', type=int, default=128, help='t embedding dimension in Q')

    # latent diffusion related parameters
    parser.add_argument('--n_interval_posterior', type=int, default=100, help='number of diffusion steps used here')
    parser.add_argument('--n_interval_prior', type=int, default=100, help='number of diffusion steps used here')
    parser.add_argument('--logsnr_min', type=float, default=-5.1, help='minimum value of logsnr') # -5.1
    parser.add_argument('--logsnr_max', type=float, default=9.8, help='maximum value of logsnr')  # 9.8
    parser.add_argument('--diffusion_residual', type=bool, default=True, help='whether treat prediction as residual in latent diffusion model')
    parser.add_argument('--var_type', type=str, default='large', help='variance type of latent diffusion')
    parser.add_argument('--Q_with_noise', type=bool, default=True, help='whether include noise during inference')
    parser.add_argument('--p_mask', type=float, default=0.2, help='probability of prior model')
    parser.add_argument('--cond_w', type=float, default=0.0, help='weight of conditional guidance')
    
    # MCMC related parameters
    parser.add_argument('--g_l_steps', type=int, default=30, help='number of langevin steps for posterior inference')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of posterior langevin')
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of posterior langevin')
    parser.add_argument('--g_llhd_sigma', type=float, default=0.1, help='sigma for G loss')
    parser.add_argument('--e_l_steps', type=int, default=60, help='number of langevin steps for prior sampling')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of prior langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of prior langevin')

    # optimizing parameters
    parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--e_lr', type=float, default=1e-4, help='learning rate for latent ebm')
    parser.add_argument('--q_lr', type=float, default=2e-4, help='learning rate for inference model Q')
    parser.add_argument('--q_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--e_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--q_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--e_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--iterations', type=int, default=1000000, help='total number of training iterations')
    parser.add_argument('--print_iter', type=int, default=20, help='number of iterations between each print')
    parser.add_argument('--plot_iter', type=int, default=1000, help='number of iterations between each plot')
    parser.add_argument('--ckpt_iter', type=int, default=50000, help='number of iterations between each ckpt saving')
    parser.add_argument('--fid_iter', type=int, default=100, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)
