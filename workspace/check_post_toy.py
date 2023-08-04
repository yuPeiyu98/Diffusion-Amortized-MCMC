# Use both diffusion model (seperate models) as prior and posterior

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pytorch_fid_wrapper as pfw
import shutil
import datetime as dt
import re
from data.dataset import CIFAR10
from src.diffusion_net import _netG_cifar10, _netG_svhn, _netG_celeba64, _netE, _netQ, _netQ_uncond, _netQ_U, _netQ_U_toy
from src.MCMC import sample_langevin_post_z_with_prior, sample_langevin_prior_z, sample_langevin_post_z_with_gaussian
from src.MCMC import gen_samples_with_diffusion_prior, calculate_fid_with_diffusion_prior, calculate_fid


torch.multiprocessing.set_sharing_strategy('file_system')

#################### training #####################################

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                w = m.weight.data
                b = m.bias.data

                m.weight.data.copy_(torch.randn_like(w) * 0.2)
                m.bias.data.copy_(torch.randn_like(b) * 0.1)

        self.net.apply(init_weights)    

    def forward(self, z):
        return self.net(z)

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

    netG = G()
    netG.cuda()

    def sample_langevin_post_z_with_mvn(
            z, x, g_l_steps, g_l_with_noise, g_l_step_size, verbose = False
        ):
        mystr = "Step/cross_entropy/recons_loss: "
  
        for i in range(g_l_steps):
            x_hat = netG(z)
            g_log_lkhd = 1.0 / (2.0 * .25 ** 2) * torch.sum((x_hat - x) ** 2)
            en = 1.0 / 2.0 * torch.sum(z**2)
            total_en = g_log_lkhd + en
            z_grad = torch.autograd.grad(total_en, z)[0]

            z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
            if g_l_with_noise:
                z.data += g_l_step_size * torch.randn_like(z)
            mystr += "{}/{:.3f}/{:.3f}/{:.8f}/{:.8f}  ".format(
                i, en.item(), g_log_lkhd.item(), 
                z.mean().item(), (z_grad - z).mean().item())
        if verbose:
            print("Log posterior sampling.")
            print(mystr)
        return z.detach()

    def plt_samples(
        samples, filename, npts=100, 
        low=-4, high=4, kde=True, kde_bw=.15
    ):
        from scipy.stats import gaussian_kde
        kernel = gaussian_kde(samples.T, bw_method=kde_bw)
        
        X, Y = np.mgrid[low:high:100j, low:high:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        fig = plt.figure(figsize=(8, 8))
        plt.xlim([low, high])
        plt.ylim([low, high])
        plt.imshow(Z, cmap='viridis', extent=[low, high, low, high])
        plt.axis('off')
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(
            fname=filename, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

    bs = 5000

    zk_pos = torch.randn(bs, 2).cuda()
    x = netG(zk_pos)
    for i in range(10):
        zk_pos.requires_grad = True
        zk_pos = sample_langevin_post_z_with_mvn(
            zk_pos, x, 100, True, 0.2, verbose = False
        )

        plt_samples(
            samples=zk_pos.cpu().detach().numpy(),
            filename='{}/{}_lang_post_toy_gt.png'.format(img_dir, i)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--log_path', type=str, default='../logs/', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../noise_mixture_nce/ncebm_torch/data', help='data path')
    parser.add_argument('--resume_path', type=str, default=None, help='pretrained ckpt path for resuming training')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--nc', type=int, default=3, help='image channel')
    parser.add_argument('--n_fid_samples', type=int, default=50000, help='number of samples for calculating fid during training')
    
    # network structure related parameters
    parser.add_argument('--nz', type=int, default=2, help='z vector length')
    parser.add_argument('--ngf', type=int, default=128, help='base channel numbers in G')
    parser.add_argument('--nif', type=int, default=64, help='base channel numbers in Q encoder')
    parser.add_argument('--nxemb', type=int, default=128, help='x embedding dimension in Q')
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
    parser.add_argument('--g_l_steps', type=int, default=50, help='number of langevin steps for posterior inference')
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
    parser.add_argument('--ckpt_iter', type=int, default=50000, help='number of iterations between each ckpt saving')
    parser.add_argument('--fid_iter', type=int, default=500, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)
