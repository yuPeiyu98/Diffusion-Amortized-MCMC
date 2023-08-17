# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:11:30 2022

@author: zhuya
"""

import os
import torch
import random
import time
import copy
import numpy as np
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt

# implement HMC sampling

######################### hyper parameters ####################################

ckpt_path = './logs_2spirals_bl/ckpt/15000.pth.tar'
h_dim = 256
n_interval = 10
n_samples = 1000000

vmin, vmax = -4, 4
nrow = 3
beta_start=0.0001 
beta_end=0.02

n_step = 20
grid = 100
step_mul = 1.02
L = 3
init_step_sz = 0.1

def get_sigma_schedule(beta_start, beta_end, num_diffusion_timesteps):
    """
    Get the noise level schedule
    :param beta_start: begin noise level
    :param beta_end: end noise level
    :param num_diffusion_timesteps: number of timesteps
    :return:
        -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
        -- a_s: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
    """
    betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
    betas = np.append(betas, 1.)
    assert isinstance(betas, np.ndarray)
    betas = betas.astype(np.float64)
    assert (betas > 0).all() and (betas <= 1).all()
    sqrt_alphas = np.sqrt(1. - betas)
    idx = np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]]).astype(np.int32)
    a_s = np.concatenate([[np.prod(sqrt_alphas[: idx[0] + 1])], np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
    sigmas = np.sqrt(1 - a_s ** 2)
    return sigmas, a_s

def generate_interval(a_s):
    a_left = np.concatenate([np.array([1.0]), a_s[:-1], np.array([0.0])])
    a_right = np.concatenate([np.array([1.0]), np.clip(a_s[1:] + a_s[1:] - a_s[:-1], a_min=0.05, a_max=None), np.array([0.0])])
    return a_left, a_right

    
def HMC(x, energy, idx, L, step_size):
    # one leap frog step of HMC
    # x: current sample
    # energy: energy function (suppose have multiple heads)
    # idx: current head
    # L: leap frog step
    # step_size: leaf frog step size
    _x = x.clone().detach()
    _x.requires_grad = True
    p0 = torch.randn_like(_x)
    grad = torch.autograd.grad(energy(_x)[:, idx].sum(), [_x])[0]
    p = p0 + 0.5 * step_size * grad
    _x.data = _x.data + step_size * p
    for i in range(L):
        grad = torch.autograd.grad(energy(_x)[:, idx].sum(), [_x])[0]
        p = p + step_size * grad
        _x.data = _x.data + step_size * p
        
    grad = torch.autograd.grad(energy(_x)[:, idx].sum(), [_x])[0]
    p = p + 0.5 * step_size * grad
    H0 = -energy(x)[:, idx] + 0.5 * torch.sum(p0 ** 2, 1)
    H1 = -energy(_x)[:, idx] + 0.5 * torch.sum(p ** 2, 1)
    p_acc = torch.minimum(torch.ones_like(H0), torch.exp(H0 - H1))
    tmp = torch.rand_like(p_acc)
    replace_idx = (p_acc > tmp)
    x[replace_idx] = _x[replace_idx].detach().clone()
    acc_rate = torch.mean(replace_idx.float()).item()
    return x, acc_rate

class network(nn.Module):
    def __init__(self, h_dim, n_class):
        super(network, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(2, h_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim, n_class - 1)
                )
    def forward(self, x):
        en = self.model(x)
        ref_q = -0.5 * torch.sum(x**2, dim=-1, keepdim=True)
        return torch.cat([en, ref_q], dim=-1)

model = network(h_dim, n_interval)
model.load_state_dict(torch.load(ckpt_path))
model.cuda()

sigmas, a_s = get_sigma_schedule(beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=n_interval-2)
a_s_cum = np.cumprod(a_s)
al, ar = generate_interval(a_s_cum)
sample_sigma = 0.5 * (al + ar)

step_sz = init_step_sz
samples = torch.randn(n_samples, 2).cuda()
all_samples = []
for i in range(n_interval - 2, -1, -1):
    start_time = time.time()
    for j in range(n_step):
        samples, acc_rate = HMC(x=samples, energy=model, idx=i, L=L, step_size=step_sz)
        if acc_rate > 0.651:
            step_sz *= step_mul
        else:
            step_sz /= step_mul
    all_samples.append(np.clip(samples.detach().clone().cpu().numpy(), a_min=vmin, a_max=vmax))
    print("Finish energy {} time {:.3f} final step size {} final accept rate {:.3f}".format(i, time.time() - start_time, step_sz, acc_rate))

fig, axes = plt.subplots(nrow, n_interval // nrow)
bins = np.linspace(vmin, vmax, grid+1)
for i in range(len(all_samples)):
    ii = i // (n_interval // nrow)
    jj = i % (n_interval // nrow) 
    tmp = all_samples[i] 
    h, _, _, _ = axes[ii, jj].hist2d(tmp[:, 0], tmp[:, 1], bins=[bins, bins], density=False)
    h = h / np.sum(h)
    h = np.transpose(h, (1, 0))
    h = np.flip(h, 0)
    axes[ii, jj].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
fig.savefig('HMC_{:.2f}_{}_{}_density_bl.png'.format(init_step_sz, n_step, L))
plt.close(fig)

fig, axes = plt.subplots(nrow, n_interval // nrow)
bins = np.linspace(vmin, vmax, grid+1)
for i in range(len(all_samples)):
    ii = i // (n_interval // nrow)
    jj = i % (n_interval // nrow) 
    tmp = all_samples[i] 
    axes[ii, jj].scatter(tmp[:, 0], tmp[:, 1], s=1)
    axes[ii, jj].set_xlim([vmin, vmax])
    axes[ii, jj].set_ylim([vmin, vmax])
fig.savefig('HMC_{:.2f}_{}_{}_sample_bl.png'.format(init_step_sz, n_step, L))
plt.close(fig)