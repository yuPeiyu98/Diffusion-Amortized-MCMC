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
from data.dataset import CIFAR10
from src.diffusion_net import _netG_cifar10, _netE, _netQ, _netQ_uncond, _netQ_U
from src.MCMC import sample_langevin_post_z_with_prior, sample_langevin_post_z_with_prior_mh, sample_langevin_post_z_with_gaussian
from src.MCMC import gen_samples_with_diffusion_prior, calculate_fid_with_diffusion_prior, calculate_fid_with_diffusion_prior_E


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
    
    img_dir = os.path.join(args.resume_path, 'imgs')
    ckpt_dir = os.path.join(args.resume_path, 'ckpt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.resume_path, osp.basename(__file__)))

    # load dataset and calculate statistics
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    # trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    # trainset = torchvision.datasets.SVHN(root=args.data_path, split='train', download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    train_iter = iter(trainloader)

    start_time = time.time()
    print("Begin calculating real image statistics")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_test)
    # testset = torchvision.datasets.SVHN(root=args.data_path, split='train', download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    mset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    # mset = torchvision.datasets.SVHN(root=args.data_path, split='test', download=True, transform=transform_test)
    mloader = data.DataLoader(mset, batch_size=500, shuffle=False, num_workers=0, drop_last=False)
    
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

    E = _netE(nz=args.nz)

    G.cuda()
    Q.cuda()
    E.cuda()
    Q_dummy.cuda()

    start_iter = 0
    fid_best = 10000
    mse_best = 10000
    save_recon_imgs = False
    if args.resume_path is not None:
        ckpt_path = os.path.join(args.resume_path, 'ckpt/best.pth.tar')
        print('load from ', ckpt_path)
        state_dict = torch.load(ckpt_path)
        G.load_state_dict(state_dict['G_state_dict'])
        Q.load_state_dict(state_dict['Q_state_dict'])
        Q_dummy.load_state_dict(state_dict['Q_dummy_state_dict'])
        E.load_state_dict(state_dict['E_state_dict'])
        
        fid_s_time = time.time()
        out_fid = calculate_fid_with_diffusion_prior(
            n_samples=args.n_fid_samples, device=x.cuda().device, netQ=Q, netG=G, netE=E,
            real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, "test"))
        print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(time.time() - fid_s_time, out_fid, fid_best))

        fid_s_time = time.time()
        out_fid = calculate_fid_with_diffusion_prior_E(
            n_samples=args.n_fid_samples, device=x.cuda().device, netQ=Q, netG=G, netE=E,
            real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, "test"))
        print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(time.time() - fid_s_time, out_fid, fid_best))

        mse_lss = 0.0
        mse_s_time = time.time()

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

            if save_recon_imgs:
                with torch.no_grad():
                    x_hat_q = G(z0)
                save_images = x[:64].detach().cpu()
                torchvision.utils.save_image(
                    torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_obs.png'.format(img_dir, iteration), normalize=True, nrow=8)
                save_images = x_hat[:64].detach().cpu()
                torchvision.utils.save_image(
                    torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_post.png'.format(img_dir, iteration), normalize=True, nrow=8)
                save_images = x_hat_q[:64].detach().cpu()
                torchvision.utils.save_image(
                    torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_post_Q.png'.format(img_dir, iteration), normalize=True, nrow=8)

            mse_lss += g_loss.item()

        mse_lss /= len(mset)
        print("Finish calculating mse time {:.3f} mse {:.3f}".format(time.time() - mse_s_time, mse_lss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_path', type=str, default='../logs/cifar', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    parser.add_argument('--resume_path', type=str, default='../logs/cifar/20230302_143524/', 
                                         help='pretrained ckpt path for resuming training')
    
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

    args = parser.parse_args()
    main(args)
