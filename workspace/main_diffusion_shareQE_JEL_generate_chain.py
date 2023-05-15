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
from src.MCMC import sample_langevin_post_z_with_prior, sample_langevin_post_z_with_gaussian
from src.MCMC import gen_samples_with_diffusion_prior, gen_samples


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
    
    img_dir = os.path.join(args.resume_path, 'imgs_chain')
    ckpt_dir = os.path.join(args.resume_path, 'ckpt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.resume_path, osp.basename(__file__)))

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
        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_test)
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
    mloader = data.DataLoader(mset, batch_size=500, shuffle=False, num_workers=0, drop_last=False)
    train_iter = iter(trainloader)

    # define models
    if args.dataset == 'cifar10':
        G = _netG_cifar10(nz=args.nz, ngf=args.ngf, nc=args.nc)
    elif args.dataset == 'svhn':
        G = _netG_svhn(nz=args.nz, ngf=args.ngf, nc=args.nc)
    elif args.dataset == 'celeba64':
        G = _netG_celeba64(nz=args.nz, ngf=args.ngf, nc=args.nc)
    elif args.dataset == 'celebaHQ':
        G = _netG_celebaHQ(nz=args.nz, ngf=args.ngf, nc=args.nc)
    Q = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_dummy = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)

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
        
        dummy = torch.randn(3,3).cuda()

        bs = 64
        fid_s_time = time.time()

        for i in range(10):
            en_l = []
            z_l = []

            z = torch.randn(bs, args.nz).cuda()
            z.requires_grad = True

            for k in range(2500):
                en = E(z).sum()
                en_b = E(z)..unsqueeze(1).detach().cpu().numpy()

                z_norm = 1.0 / 2.0 * torch.sum(z**2)
                z_grad = torch.autograd.grad(en + z_norm, z)[0]

                z.data = z.data - 0.5 * args.e_l_step_size * args.e_l_step_size * z_grad 
                if True:
                    z.data += args.e_l_step_size * torch.randn_like(z)

                if (k % 100 == 0 or k == args.e_l_steps - 1):
                    en_l.append(en_b)
                    z_l.append(z)

            for t, zk_prior in enumerate(z_l):
                zk_prior = zk_prior.clone().detach()
                with torch.no_grad():
                    x = G(zk_prior)
                cur_samples = x
                fid_samples = (1.0 + torch.clamp(cur_samples, min=-1.0, max=1.0)) / 2.0
                for j, sample in enumerate(fid_samples):
                    torchvision.utils.save_image(
                        sample, '{}/fid_chain_{:05d}_{:04d}.png'.format(img_dir, i * bs + j, t), 
                        normalize=True)

            en_l = np.hstack(en_l)
            np.save('{}/en_{:04d}.npy'.format(img_dir, i), en_l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--log_path', type=str, default='../logs/cifar', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    parser.add_argument('--resume_path', type=str, default='../logs/svhn/20230304_204909/', 
                                         help='pretrained ckpt path for resuming training')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--nc', type=int, default=3, help='image channel')
    parser.add_argument('--n_fid_samples', type=int, default=50000, help='number of samples for calculating fid during training')
    
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
    parser.add_argument('--e_l_steps', type=int, default=100, help='number of langevin steps for prior sampling')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of prior langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of prior langevin')

    args = parser.parse_args()
    main(args)
