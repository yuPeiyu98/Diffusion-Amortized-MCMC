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
import sklearn
import sklearn.datasets
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.utils import shuffle as util_shuffle

# implement parallel tempering based on HMC sampling

######################### hyper parameters ####################################
data_type = "2spirals"
baseline = False
if baseline:
    ckpt_path = './logs_{}_bl/ckpt/15000.pth.tar'.format(data_type)
else:
    ckpt_path = './logs_{}/ckpt/15000.pth.tar'.format(data_type)
    
h_dim = 256
n_interval = 10 
n_samples = 1000000

vmin, vmax = -4, 4
nrow = 3
beta_start=0.0001 
beta_end=0.02

n_step = 151
grid = 100
step_mul = 1.02
L = 3
init_step_sz = 0.1
n_explore = 10
n_print = 10
n_img = 10
log_dir = './pt_sampling'

def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.array(centers)
        '''
        dataset = []
        
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] = point[0] + center[0]
            point[1] = point[1] + center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        '''
        point = rng.randn(batch_size, 2) * 0.5
        idx = rng.randint(low = 0, high=8, size=batch_size)
        center = centers[idx]
        dataset = point + center
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] = features[:, 0] + 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((d1x, d1y)))) / 3
        x = x + np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", rng, batch_size)

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

def noise_injection(x, left, right, baseline=False):
    assert len(x) == len(left)
    assert len(left) == len(right)
    nsamples = len(x)
    noise = np.random.randn(nsamples, 2)
    if baseline:
        #print('using only middle point')
        a_sample =  np.expand_dims((right + left) / 2.0, -1)
    else:
        a_sample = np.expand_dims(np.random.uniform(low=right, high=left), -1)
    assert (a_sample >=0.0).all()
    x_noise = x * a_sample + noise * np.sqrt(1.0 - a_sample**2)
    return x_noise

    
def HMC(x, energy, idx, L, step_size):
    """
    # one leap frog step of HMC
    # x: current sample
    # energy: energy function (suppose have multiple heads)
    # idx: current head
    # L: leap frog step
    # step_size: leaf frog step size
    """
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

os.makedirs(log_dir, exist_ok=True)

model = network(h_dim, n_interval)
model.load_state_dict(torch.load(ckpt_path))
model.cuda()

sigmas, a_s = get_sigma_schedule(beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=n_interval-2)
a_s_cum = np.cumprod(a_s)
al, ar = generate_interval(a_s_cum)



# estimate true distribution
num_samples = 1000000
bins = np.linspace(vmin, vmax, grid+1)
true_dist = []

for i in range(n_interval):
    true_samples = inf_train_gen(data_type, batch_size=num_samples)
    idx = np.ones(num_samples, dtype=np.int) * i
    left = al[idx]
    right = ar[idx]
    noise_samples = noise_injection(true_samples, left, right, baseline=baseline)
    noise_samples = np.clip(noise_samples, a_min=vmin, a_max=vmax)
    h, _, _, _ = plt.hist2d(noise_samples[:, 0], noise_samples[:, 1], bins=[bins, bins], density=False)
    h = h / np.sum(h)
    h = np.transpose(h, (1, 0))
    h = np.flip(h, 0)
    true_dist.append(copy.deepcopy(h))

step_szs = [init_step_sz for _ in range(n_interval)] 
acc_rates = [1.0 for _ in range(n_interval)]
all_samples = torch.randn(n_interval, n_samples, 2).cuda()

start_time = time.time()
mse = [[] for _ in range(n_interval)]
flag = 0
for j in range(n_step):
    for i in range(n_interval):
        tmp_samples, acc_rate = HMC(x=all_samples[i], energy=model, idx=i, L=L, step_size=step_szs[i])
        all_samples[i] = tmp_samples.detach().clone()
        acc_rates[i] = acc_rate
        if acc_rate > 0.651:
            step_szs[i] = step_szs[i] * step_mul
        else:
            step_szs[i] = step_szs[i] / step_mul
            
    if j > 0 and j % n_explore == 0:
        for i in range(n_interval // 2):
            idx1 = 2 * i + flag
            idx2 = idx1 + 1
            if idx1 > n_interval - 1 or idx2 > n_interval - 1:
                continue
            x1 = all_samples[idx1].detach().clone()
            x2 = all_samples[idx2].detach().clone()
            with torch.no_grad():
                energies1 = model(x1)
                energies2 = model(x2)
                pt_acc = torch.minimum(torch.ones_like(energies1[:, idx2]), torch.exp(energies1[:, idx2] + energies2[:, idx1] - energies1[:, idx1] - energies2[:, idx2]))
                tmp = torch.rand_like(pt_acc)
                exchange_idx = (pt_acc > tmp)
                all_samples[idx1, exchange_idx] = x2[exchange_idx].detach().clone()
                all_samples[idx2, exchange_idx] = x1[exchange_idx].detach().clone()
            
        flag = (flag + 1) % 2
    
    if j % n_print == 0:
        print("Finish {} time {:.2f} ---------------------------------".format(j, time.time() - start_time))
        for i in range(n_interval):
            print("Energy {} step size {:.3f} accept rate {:.3f}".format(i, step_szs[i], acc_rates[i]))
    
    _all_samples = np.clip(all_samples.detach().clone().cpu().numpy(), a_min=vmin, a_max=vmax)
    fig, axes = plt.subplots(nrow, n_interval // nrow)
    for i in range(len(_all_samples) - 1):
        ii = i // (n_interval // nrow)
        jj = i % (n_interval // nrow) 
        tmp = _all_samples[i] 
        h, _, _, _ = axes[ii, jj].hist2d(tmp[:, 0], tmp[:, 1], bins=[bins, bins], density=False)
        h = h / np.sum(h)
        h = np.transpose(h, (1, 0))
        h = np.flip(h, 0)
        mse[i].append(np.sum((true_dist[i] - h) ** 2))
        axes[ii, jj].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
        
    if j % n_img == 0:
        fig.savefig(os.path.join(log_dir, 'PT_{}.png'.format(j)))
        np.save(os.path.join(log_dir, 'mse.npy'), np.array(mse))
    plt.close(fig)
    
    fig, axes = plt.subplots(nrow, n_interval // nrow)    
    for i in range(n_interval - 1):
        ii = i // (n_interval // nrow)
        jj = i % (n_interval // nrow)   
        axes[ii, jj].plot(np.arange(len(mse[i])), mse[i])
    fig.savefig(os.path.join(log_dir, 'mse.png'))
    plt.close(fig)