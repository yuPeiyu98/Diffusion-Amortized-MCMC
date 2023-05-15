# Use both diffusion model (seperate models) as prior and posterior

import argparse
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
from data.dataset import LSUN
from src.stylegan.stylegan_generator import StyleGANGenerator
from src.stylegan.stylegan_encoder import StyleGANEncoder
from src.stylegan.perceptual_model import PerceptualModel
from src.diffusion_net_stylegan import _netE, _netQ_uncond, _netQ_U
from src.MCMC import sample_langevin_post_z_with_prior, sample_langevin_prior_z, sample_langevin_post_z_with_gaussian
from src.MCMC import sample_invert_z, gen_samples_with_diffusion_prior_stylegan, calculate_fid_with_samples

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
    if args.dataset == 'ffhq':
        args.nz = 7168
        args.nxemb = 7168
        args.pretrained_G_path = osp.join(args.pretrained_G_path, 'styleganinv_ffhq256_generator.pth')
        args.pretrained_E_path = osp.join(args.pretrained_E_path, 'styleganinv_ffhq256_encoder.pth')
        args.data_path = osp.join(args.data_path, 'ffhq')
        trainset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'ffhq_train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'ffhq_test'), transform=transform_test) 
        mset = torchvision.datasets.ImageFolder(root=osp.join(args.data_path, 'ffhq_test'), transform=transform_test)
    elif args.dataset == 'lsun_tower':
        args.nz = 7168
        args.nxemb = 7168
        args.pretrained_G_path = osp.join(args.pretrained_G_path, 'styleganinv_tower256_generator.pth')
        args.pretrained_E_path = osp.join(args.pretrained_E_path, 'styleganinv_tower256_encoder.pth')
        args.data_path = osp.join(args.data_path, 'lsun')
        trainset = LSUN(root=args.data_path, classes=['tower_train'], transform=transform_train)
        testset = LSUN(root=args.data_path, classes=['tower_val'], transform=transform_test) 
        mset = LSUN(root=args.data_path, classes=['tower_val'], transform=transform_test)
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

    G = StyleGANGenerator(weight_path=args.pretrained_G_path)
    Q = StyleGANEncoder(weight_path=args.pretrained_E_path, load=False, resolution=256)

    E = _netE(nz=args.nz, e_sn=False)
    F = PerceptualModel(weight_path=args.pretrained_F_path)

    G.cuda()
    Q.cuda()
    E.cuda()
    F.cuda()

    G.eval()
    E.eval()
    F.eval()

    Q_optimizer = optim.Adam(Q.parameters(), lr=args.q_lr, betas=(0.9, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=args.e_lr, betas=(0.9, 0.999))

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

        Q.train()
        
        z = Q(x)
        
        Q_loss = torch.mean((G(z) - x) ** 2) + torch.mean((netF(x) - netF(x_hat)) ** 2, dim=[1,2,3]).mean() * 5e-5 
        Q_loss.backward()
        if args.q_is_grad_clamp:
            torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=args.q_max_norm)
        
        # update Q 
        Q_optimizer.step()
        Q_optimizer.zero_grad()

        Q.eval()

        # learning rate schedule
        if (iteration + 1) % 1000 == 0:
            q_lr = max(q_lr * 0.99, 1e-5)
            e_lr = max(e_lr * 0.99, 1e-5)
            for Q_param_group in Q_optimizer.param_groups:
                Q_param_group['lr'] = q_lr
            for E_param_group in E_optimizer.param_groups:
                E_param_group['lr'] = e_lr

        if iteration % args.print_iter == 0:
            print("Iter {} time {:.2f} g_loss {:.6f} q_loss {:.3f} e_loss {:.3f} e_pos {:.3f} e_neg {:.3f} q_lr {:.8f}".format(
                iteration, time.time() - start_time, g_loss.item(), Q_loss.item(), 
                E_loss.item(), e_pos.mean().item(), e_neg.mean().item(), q_lr))
            print(zk_pos.max(), zk_pos.min())
        
        if iteration > 0 and iteration % args.ckpt_iter == 0:
            print('Saving checkpoint')
            save_dict = {
                'Q_state_dict': Q.state_dict(),
                'Q_optimizer': Q_optimizer.state_dict(),
                'E_state_dict': E.state_dict(),
                'E_optimizer': E_optimizer.state_dict(),
                'iter': iteration
            }
            torch.save(save_dict, os.path.join(ckpt_dir, '{}.pth.tar'.format(iteration)))
        
        if iteration % args.fid_iter == 0:
            mse_lss = 0.0
            mse_s_time = time.time()

            i = 0

            samples = []
            for x, _ in mloader:
                x = x.cuda()
                with torch.no_grad():
                    z0 = Q(x)
                zk_pos = z0.detach().clone()
                zk_pos.requires_grad = True
                zk_pos = sample_invert_z(
                            z=zk_pos, x=x, netG=G, netF=F, netE=E,
                            g_l_steps=args.g_l_steps, g_l_step_size=args.g_l_step_size, 
                            verbose = False)

                with torch.no_grad():
                    x_hat = G(zk_pos)
                    g_loss = torch.mean((x_hat - x) ** 2, dim=[1, 2, 3]).sum()
                mse_lss += g_loss.item()

                samples.append(x_hat.detach().clone())

            mse_lss /= len(mset)
            if mse_lss < mse_best:
                mse_best = mse_lss
            print("Finish calculating mse time {:.3f} mse {:.3f} / {:.3f}".format(time.time() - mse_s_time, mse_lss, mse_best))

            fid_s_time = time.time()
            out_fid = calculate_fid_with_samples(
                fid_samples=samples,
                real_m=real_m, real_s=real_s, save_name='{}/fid_samples_{}.png'.format(img_dir, iteration))
            if out_fid < fid_best:
                fid_best = out_fid
                print('Saving best checkpoint')
                save_dict = {
                    'Q_state_dict': Q.state_dict(),
                    'Q_optimizer': Q_optimizer.state_dict(),
                    'E_state_dict': E.state_dict(),
                    'E_optimizer': E_optimizer.state_dict(),
                    'iter': iteration
                }
                torch.save(save_dict, os.path.join(ckpt_dir, 'best.pth.tar'))
            print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(time.time() - fid_s_time, out_fid, fid_best))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', type=str, default='lsun_tower')
    parser.add_argument('--log_path', type=str, default='../logs/', help='log directory')
    parser.add_argument('--data_path', type=str, default='../../../datasets/', help='data path')
    parser.add_argument('--resume_path', type=str, default=None, help='pretrained ckpt path for resuming training')
    parser.add_argument('--pretrained_G_path', type=str, default='../../idinvert/', 
                                                         help='pretrained ckpt path for generator')
    parser.add_argument('--pretrained_E_path', type=str, default='../../idinvert/', 
                                                         help='pretrained ckpt path for generator')
    parser.add_argument('--pretrained_F_path', type=str, default='../../idinvert/vgg16.pth', 
                                                         help='pretrained ckpt path for perceptual model')
    
    # data related parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--nc', type=int, default=3, help='image channel')
    parser.add_argument('--n_fid_samples', type=int, default=50000, help='number of samples for calculating fid during training')
    
    # network structure related parameters
    parser.add_argument('--nz', type=int, default=128, help='z vector length')
    parser.add_argument('--ngf', type=int, default=128, help='base channel numbers in G')
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
    parser.add_argument('--g_l_steps', type=int, default=100, help='number of langevin steps for posterior inference')
    parser.add_argument('--g_l_step_size', type=float, default=0.01, help='stepsize of posterior langevin')
    parser.add_argument('--g_l_with_noise', default=False, type=bool, help='noise term of posterior langevin')
    parser.add_argument('--g_llhd_sigma', type=float, default=1, help='sigma for G loss')
    parser.add_argument('--e_l_steps', type=int, default=60, help='number of langevin steps for prior sampling')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of prior langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of prior langevin')

    # optimizing parameters
    parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--e_lr', type=float, default=5e-4, help='learning rate for latent ebm')
    parser.add_argument('--q_lr', type=float, default=1e-4, help='learning rate for inference model Q')
    parser.add_argument('--q_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--e_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=True, help='whether doing the gradient clamp')
    parser.add_argument('--q_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--e_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--iterations', type=int, default=1000000, help='total number of training iterations')
    parser.add_argument('--print_iter', type=int, default=1, help='number of iterations between each print')
    parser.add_argument('--plot_iter', type=int, default=1000, help='number of iterations between each plot')
    parser.add_argument('--ckpt_iter', type=int, default=100, help='number of iterations between each ckpt saving')
    parser.add_argument('--fid_iter', type=int, default=15, help='number of iterations between each fid computation')

    args = parser.parse_args()
    main(args)