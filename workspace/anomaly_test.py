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
from src.diffusion_net import _netG_mnist, _netE, _netQ, _netQ_uncond, _netQ_U
from src.MCMC import sample_langevin_post_z_with_prior, sample_langevin_post_z_with_prior_mh, sample_langevin_post_z_with_gaussian
from src.MCMC import gen_samples_with_diffusion_prior, calculate_fid_with_diffusion_prior

from sklearn.metrics import roc_curve, precision_recall_curve, auc

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
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    testset = MNIST(root=osp.join(args.data_path), split='test', label=args.label, transform=transform_test, load=True) 
    testloader = data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=0, drop_last=False)

    # define models
    G = _netG_mnist(nz=args.nz, ngf=args.ngf, nc=args.nc)
    Q = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_dummy = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_eval = _netQ_U(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
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
    Q_optimizer = optim.AdamW(Q.parameters(), weight_decay=1e-2, lr=args.q_lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=args.e_lr, betas=(0.5, 0.999))

    start_iter = 0
    auc_best = 0.0
    if args.resume_path is not None:
        ckpt_path = os.path.join(args.resume_path, 'ckpt/best.pth.tar')
        print('load from ', ckpt_path)
        state_dict = torch.load(ckpt_path)
        G.load_state_dict(state_dict['G_state_dict'])
        Q.load_state_dict(state_dict['Q_state_dict'])
        Q_dummy.load_state_dict(state_dict['Q_dummy_state_dict'])
        E.load_state_dict(state_dict['E_state_dict'])

    eval_time = time.time()

    s_list = []
    l_list = []
    for x, l in testloader:
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
            s_hat = E(zk_pos)
            score = torch.sum((x_hat - x) ** 2, dim=[1, 2, 3]) * 1 + s_hat + torch.sum(zk_pos ** 2, dim=-1) * 0.5
            s_list.append(score)
            l_list.append(l)

    l_arr = torch.cat(l_list, dim=0).detach().cpu().numpy()
    s_arr = torch.cat(s_list, dim=0).detach().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(l_arr, s_arr)
    prc_auc = auc(recall, precision)
            
    print("Finish calculating auc time {:.3f} auc {:.3f} / {:.3f}".format(time.time() - eval_time, prc_auc, auc_best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--log_path', type=str, default='../logs/', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data/mnist', help='data path')
    parser.add_argument('--resume_path', type=str, default='../logs/mnist/20230401_192026/', 
                                         help='pretrained ckpt path for resuming training')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--nc', type=int, default=1, help='image channel')
    parser.add_argument('--label', type=int, default=4, help='held out digit')
    
    # network structure related parameters
    parser.add_argument('--nz', type=int, default=100, help='z vector length')
    parser.add_argument('--ngf', type=int, default=128, help='base channel numbers in G')
    parser.add_argument('--nif', type=int, default=128, help='base channel numbers in Q encoder')
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
    parser.add_argument('--print_iter', type=int, default=100, help='number of iterations between each print')
    parser.add_argument('--plot_iter', type=int, default=1000, help='number of iterations between each plot')
    parser.add_argument('--ckpt_iter', type=int, default=1000, help='number of iterations between each ckpt saving')
    parser.add_argument('--eval_iter', type=int, default=500, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)
