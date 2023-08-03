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
from src.diffusion_net import _netG_cifar10, _netG_svhn, _netG_celeba64, _netE, _netQ, _netQ_uncond, _netQ_U, _netQ_U_toy
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

    Q = _netQ_U_toy(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_dummy = _netQ_U_toy(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    Q_eval = _netQ_U_toy(nc=args.nc, nz=args.nz, nxemb=args.nxemb, ntemb=args.ntemb, nif=args.nif, \
        diffusion_residual=args.diffusion_residual, n_interval=args.n_interval_posterior, 
        logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max, var_type=args.var_type, with_noise=args.Q_with_noise, cond_w=args.cond_w,
        net_arch='A', dataset=args.dataset)
    for param, target_param in zip(Q.parameters(), Q_dummy.parameters()):
        target_param.data.copy_(param.data)
    for param, target_param in zip(Q.parameters(), Q_eval.parameters()):
        target_param.data.copy_(param.data)

    Q.cuda()
    Q_dummy.cuda()
    Q_eval.cuda()

    Q_optimizer = optim.AdamW(Q.parameters(), weight_decay=1e-2, lr=args.q_lr, betas=(0.5, 0.999))
    # E_optimizer = optim.Adam(E.parameters(), lr=args.e_lr, betas=(0.5, 0.999))

    start_iter = 0
    fid_best = 10000
    mse_best = 10000
    if args.resume_path is not None:
        print('load from ', args.resume_path)
        state_dict = torch.load(args.resume_path)        
        Q.load_state_dict(state_dict['Q_state_dict'])        
        Q_optimizer.load_state_dict(state_dict['Q_optimizer'])
        start_iter = state_dict['iter'] + 1
    
    q_lr = args.q_lr
    e_lr = args.e_lr
    rho = 0.005
    p_mask = args.p_mask

    S = np.array([
        [0.7, 0.6],
        [0.7, 0.8]
    ])
    S_inv = np.linalg.inv(S)
    L = np.linalg.cholesky(S)

    # (1, 2, 2)
    S_inv_torch = torch.tensor(S_inv).float().unsqueeze(0).cuda()
    L_torch = torch.tensor(L).float().unsqueeze(0).cuda()

    def G(z):
        # z = torch.randn(bs, 2)
        return z + torch.matmul(L_torch, z.unsqueeze(2)).squeeze(2)

    def log_p_x_z(z, mu):
        # z = torch.randn(bs, 2)

        # (bs, 1, 2)
        m0 = torch.matmul((z - mu).unsqueeze(1), S_inv_torch)
        # (bs, 1, 1)
        log_x_z = torch.matmul(m0, (z - mu).unsqueeze(2))
        return .5 * log_x_z.reshape(-1)

    def sample_langevin_post_z_with_mvn(
            z, x, g_l_steps, g_l_with_noise, g_l_step_size, verbose = False
        ):
        mystr = "Step/cross_entropy/recons_loss: "
  
        for i in range(g_l_steps):
            g_log_lkhd = log_p_x_z(z, x).sum()
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

    start_time = time.time()

    # begin training
    bs = 500
    for iteration in range(start_iter, args.iterations + 1):
        z = torch.randn(bs, 2).cuda()
        x = G(z).detach()

        z_mask_prob = torch.rand((len(x),), device=x.device)
        z_mask = torch.ones(len(x), device=x.device)
        z_mask[z_mask_prob < p_mask] = 0.0
        z_mask = z_mask.unsqueeze(-1)

        Q.eval()
        # infer z from given x
        with torch.no_grad():
            z0 = Q_dummy(x)
            zp = Q(x=None, b=x.size(0), device=x.device)
        zk_pos, zk_neg = z0.detach().clone(), z0.detach().clone()
        zk_pos.requires_grad = True
        zk_neg.requires_grad = True

        zk_pos = sample_langevin_post_z_with_mvn(
            z=zk_pos, x=x, g_l_steps=args.g_l_steps, g_l_with_noise=args.g_l_with_noise,
            g_l_step_size=args.g_l_step_size, verbose = (iteration % (args.print_iter * 10) == 0)
        )
        
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

        Q.eval()
        
        # learning rate schedule
        if (iteration + 1) % 1000 == 0:
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
            print("Iter {} time {:.2f} g_loss {:.6f} q_loss {:.3f} g_lr {:.8f} q_lr {:.8f}".format(
                iteration, time.time() - start_time, 0, Q_loss.item(), 0, q_lr))
            print(zk_pos.max(), zk_pos.min())
        
        if iteration > 0 and iteration % args.ckpt_iter == 0:
            print('Saving checkpoint')
            save_dict = {
                'Q_state_dict': Q.state_dict(),
                'Q_optimizer': Q_optimizer.state_dict(),
                'Q_dummy_state_dict': Q_dummy.state_dict(),
                'Q_eval_state_dict': Q_eval.state_dict(),
                'iter': iteration
            }
            torch.save(save_dict, os.path.join(ckpt_dir, '{}.pth.tar'.format(iteration)))
        
        if iteration % args.fid_iter == 0:
            bs = 500

            z_list = []
            for i in range(10):
                z = torch.randn(bs, 2).cuda()
                x = G(z).detach()

                with torch.no_grad():
                    z0 = Q(x)
                zk_pos = z0.detach().clone()
                zk_pos.requires_grad = True
                zk_pos = sample_langevin_post_z_with_mvn(
                            z=zk_pos, x=x, g_l_steps=10, # if out_fid > fid_best else 40, 
                            g_l_with_noise=True,
                            g_l_step_size=args.g_l_step_size, verbose=False
                        )

                z_list.append(zk_pos)

            z_ = torch.cat(z_list, dim=0).cpu().detach().numpy()

            low, high = -4, 4
            fig = plt.figure(figsize=(8, 8))
            plt.xlim([low, high])
            plt.ylim([low, high])
            plt.scatter(z_[:, 0], z_[:, 1], range=[[low, high], [low, high]])
            plt.axis('off')
            plt.gcf().set_size_inches(8, 8)
            plt.savefig(
                fname='{}/{}_post_Q.png'.format(img_dir, iteration), 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=100
            )
            plt.close()


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
    parser.add_argument('--ckpt_iter', type=int, default=50000, help='number of iterations between each ckpt saving')
    parser.add_argument('--fid_iter', type=int, default=500, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)
