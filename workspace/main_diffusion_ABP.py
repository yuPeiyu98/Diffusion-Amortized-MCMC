import argparse
import numpy as np
import os
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
from src.diffusion_net import _netG_cifar10, _netE, _netQ
from src.MCMC import sample_langevin_prior_z, sample_langevin_post_z, gen_samples, calculate_fid

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
    shutil.copyfile(__file__, os.path.join(args.log_path, timestamp, __file__))

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
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    
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
    E = _netE(nz=args.nz, ndf=args.nef, nez=args.nez, e_sn=args.e_sn)
    Q = _netQ(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval, logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type)
    G.cuda()
    E.cuda()
    Q.cuda()

    G_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=args.e_lr, betas=(0.5, 0.999))
    Q_optimizer = optim.Adam(Q.parameters(), lr=args.q_lr, betas=(0.5, 0.999))

    start_iter = 0
    fid_best = 10000
    if args.resume_path is not None:
        print('load from ', args.resume_path)
        state_dict = torch.load(args.resume_path)
        G.load_state_dict(state_dict['G_state_dict'])
        E.load_state_dict(state_dict['E_state_dict'])
        Q.load_state_dict(state_dict['Q_state_dict'])
        G_optimizer.load_state_dict(state_dict['G_optimizer'])
        E_optimizer.load_state_dict(state_dict['E_optimizer'])
        Q_optimizer.load_state_dict(state_dict['Q_optimizer'])
        start_iter = state_dict['iter'] + 1
    
    # begin training
    for iter in range(start_iter, args.iterations + 1):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            x, _ = next(train_iter)
        x = x.cuda()

        Q.eval()
        G.eval()
        E.eval()
        # infer z from given x
        with torch.no_grad():
            z0 = Q(x)
        zk_pos = z0.detach().clone()
        zk_pos.requires_grad = True
        zk_pos = sample_langevin_post_z(z=zk_pos, x=x, netG=G, netE=E, g_l_steps=args.g_l_steps, g_llhd_sigma=args.g_llhd_sigma, g_l_with_noise=args.g_l_with_noise, \
            g_l_step_size=args.g_l_step_size, verbose = (iter % args.print_iter == 0))

        # update Q 
        Q_optimizer.zero_grad()
        Q.train()
        Q_loss = Q.calculate_loss(x=x, z=zk_pos).mean()
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm(Q.parameters(), max_norm=1.0)
        Q_optimizer.step()
        
        # update G
        G_optimizer.zero_grad()
        G.train()
        x_hat = G(zk_pos)
        g_loss = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * torch.sum((x_hat - x) ** 2, dim=[1,2,3]).mean()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm(G.parameters(), max_norm=1.0)
        G_optimizer.step()

        # update E
        E_optimizer.zero_grad()
        E.train()
        en_pos = E(zk_pos).mean()
        E.eval()
        zk_prior = torch.randn_like(zk_pos).detach()
        zk_prior.requires_grad = True
        zk_prior = sample_langevin_prior_z(z=zk_prior, netE=E, e_l_steps=args.e_l_steps, e_l_step_size=args.e_l_step_size, e_l_with_noise=args.e_l_with_noise,\
             verbose = (iter % args.print_iter == 0))
        en_neg = E(zk_prior).mean() 
        en_loss = en_neg - en_pos
        en_loss.backward()
        torch.nn.utils.clip_grad_norm(E.parameters(), max_norm=1.0)
        E_optimizer.step()

        if iter % args.print_iter == 0:
            print("Iter {} time {:.2f} g_loss {:.2f} q_loss {:.4f} en_pos {:.2f} en_neg {:.2f} en_loss {:.3f}"\
                .format(iter, time.time() - start_time, g_loss.item(), Q_loss.item(), en_pos.item(), en_neg.item(), en_loss.item()))
        
        if iter % args.plot_iter == 0:
            # reconstruction
            with torch.no_grad():
                x_hat_q = G(z0)
                save_images = x[:64].detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_obs.png'.format(img_dir, iter), normalize=True, nrow=8)
                save_images = x_hat[:64].detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_post.png'.format(img_dir, iter), normalize=True, nrow=8)
                save_images = x_hat_q[:64].detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_post_Q.png'.format(img_dir, iter), normalize=True, nrow=8)
            # samples
            G.eval()
            E.eval()
            samples = gen_samples(bs=64, nz=args.nz, netE=E, netG=G, e_l_steps=args.e_l_steps, e_l_step_size=args.e_l_step_size, e_l_with_noise=args.e_l_with_noise) 
            save_images = samples[:64].detach().cpu()
            torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_prior.png'.format(img_dir, iter), normalize=True, nrow=8)
        
        if iter > 0 and iter % args.ckpt_iter == 0:
            print('Saving checkpoint')
            save_dict = {
                'G_state_dict': G.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'E_state_dict': E.state_dict(),
                'E_optimizer': E_optimizer.state_dict(),
                'Q_state_dict': Q.state_dict(),
                'Q_optimizer': Q_optimizer.state_dict(),
                'iter': iter
            }
            torch.save(save_dict, os.path.join(ckpt_dir, '{}.pth.tar'.format(iter)))
        
        if iter > 0 and iter % args.fid_iter == 0:
            G.eval()
            E.eval()
            fid_s_time = time.time()
            out_fid = calculate_fid(n_samples=args.n_fid_samples, nz=args.nz, netE=E, netG=G, e_l_steps=args.e_l_steps, e_l_step_size=args.e_l_step_size, \
                e_l_with_noise=args.e_l_with_noise, real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, iter))
            if out_fid < fid_best:
                fid_best = out_fid
                print('Saving best checkpoint')
                save_dict = {
                    'G_state_dict': G.state_dict(),
                    'G_optimizer': G_optimizer.state_dict(),
                    'E_state_dict': E.state_dict(),
                    'E_optimizer': E_optimizer.state_dict(),
                    'Q_state_dict': Q.state_dict(),
                    'Q_optimizer': Q_optimizer.state_dict(),
                    'iter': iter
                }
                torch.save(save_dict, os.path.join(ckpt_dir, 'best.pth.tar'))
            print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(time.time() - fid_s_time, out_fid, fid_best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_path', type=str, default='./logs/cifar', help='log directory')
    parser.add_argument('--data_path', type=str, default='./data/cifar', help='data path')
    parser.add_argument('--resume_path', type=str, default=None, help='pretrained ckpt path for resuming training')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--nc', type=int, default=3, help='image channel')
    parser.add_argument('--n_fid_samples', type=int, default=5000, help='number of samples for calculating fid during training')
    
    # network structure related parameters
    parser.add_argument('--nz', type=int, default=128, help='z vector length')
    parser.add_argument('--ngf', type=int, default=128, help='base channel numbers in G')
    parser.add_argument('--nef', type=int, default=200, help='feature dimension of latent ebm')
    parser.add_argument('--nif', type=int, default=64, help='base channel numbers in Q encoder')
    parser.add_argument('--nez', type=int, default=1, help='final output number of ebm')
    parser.add_argument('--e_sn', tyep=bool, default=False, help='whether use spectral_norm in ebm')
    parser.add_argument('--nxemb', type=int, default=128, help='x embedding dimension in Q')
    parser.add_argument('--ntemb', type=int, default=128, help='t embedding dimension in Q')

    # latent diffusion related parameters
    parser.add_argument('--n_interval', type=int, default=20, help='number of diffusion steps used here')
    parser.add_argument('--logsnr_min', type=float, default=-20., help='minimum value of logsnr')
    parser.add_argument('--logsnr_max', type=float, default=20., help='maximum value of logsnr')
    parser.add_argument('--diffusion_residual', tyep=bool, default=False, help='whether treat prediction as residual in latent diffusion model')
    parser.add_argument('--var_type', type=str, default='small', help='variance type of latent diffusion')
    
    # MCMC related parameters
    parser.add_argument('--g_l_steps', type=int, default=10, help='number of langevin steps for posterior inference')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of posterior langevin')
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of posterior langevin')
    parser.add_argument('--g_llhd_sigma', type=float, default=1.0, help='sigma for G loss')
    parser.add_argument('--e_l_steps', type=int, default=60, help='number of langevin steps for prior sampling')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of prior langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of prior langevin')

    # optimizing parameters
    parser.add_argument('--g_lr', type=float, default=5e-4, help='learning rate for generator')
    parser.add_argument('--e_lr', type=float, default=1e-5, help='learning rate for latent ebm')
    parser.add_argument('--q_lr', type=float, default=1e-4, help='learning rate for inference model Q')
    parser.add_argument('--iterations', type=int, default=1000000, help='total number of training iterations')
    parser.add_argument('--print_iter', tyep=int, default=100, help='number of iterations between each print')
    parser.add_argument('--plot_iter', tyep=int, default=1000, help='number of iterations between each plot')
    parser.add_argument('--ckpt_iter', tyep=int, default=50000, help='number of iterations between each ckpt saving')
    parser.add_argument('--fid_iter', tyep=int, default=25000, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)
