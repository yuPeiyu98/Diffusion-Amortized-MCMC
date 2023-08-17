# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:43:43 2022

@author: zhuya
"""
# add caculating mse

import numpy as np
import os
import torch
import random
import time
import copy
import torch.optim as optim
import sklearn
import sklearn.datasets
import torch.nn as nn
from sklearn.utils import shuffle as util_shuffle
from matplotlib import pyplot as plt

################################## generate data ##############################
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

######################### define neural network ###############################
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
    
######################## noise injection function #############################
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
########################## hyper parameters ###################################
baseline = True

batch_size = 30000
data_type = "2spirals"

h_dim = 256
lr = 1e-4
iterations = 15001

n_interval = 10
beta_start=0.0001 
beta_end=0.02

log_path = './logs_' + data_type 
if baseline:
    log_path += '_bl'
print_iter = 250
moniter_iter = 250
plot_iter = 2500
ckpt_iter = 25000

grid = 300 # number of cells to plot
vmin, vmax = -4, 4
nrow = 2 # number of plot in one line
########################## training loop ######################################
def train():
    img_dir = os.path.join(log_path, 'imgs')
    ckpt_dir = os.path.join(log_path, 'ckpt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # generate intervals
    sigmas, a_s = get_sigma_schedule(beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=n_interval-2)
    a_s_cum = np.cumprod(a_s)
    al, ar = generate_interval(a_s_cum)
    
    plt.figure()
    plt.plot(np.arange(len(al)), al - ar)
    plt.savefig(os.path.join(img_dir, 'interval_length.png'))
    plt.close()
    
    # estimate true distribution
    fig, axes = plt.subplots(nrow, n_interval // nrow)
    num_samples = 10000000
    bins = np.linspace(vmin, vmax, grid+1)
    true_dist = []

    for i in range(n_interval):
        true_samples = inf_train_gen(data_type, batch_size=num_samples)
        idx = np.ones(num_samples, dtype=np.int) * i
        left = al[idx]
        right = ar[idx]
        noise_samples = noise_injection(true_samples, left, right, baseline=baseline)
        noise_samples = np.clip(noise_samples, a_min=vmin, a_max=vmax)
        ii = i // (n_interval // nrow)
        jj = i % (n_interval // nrow)
        h, _, _, _ = axes[ii, jj].hist2d(noise_samples[:, 0], noise_samples[:, 1], bins=[bins, bins], density=False)
        h = h / np.sum(h)
        h = np.transpose(h, (1, 0))
        h = np.flip(h, 0)
        true_dist.append(copy.deepcopy(h))
        axes[ii, jj].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    fig.savefig(os.path.join(img_dir, 'true_dist.png'))
    plt.close(fig)

    fig, axes = plt.subplots(nrow, n_interval // nrow)
    for i in range(n_interval):
        ii = i // (n_interval // nrow)
        jj = i % (n_interval // nrow) 
        tmp = true_dist[i] 
        axes[ii, jj].matshow(tmp)
        axes[ii, jj].tick_params(left = False, right = False, labelleft = False, labelbottom = False, labeltop = False, bottom = False, top=False)
    fig.savefig(os.path.join(img_dir, 'true_dist_h.png'))
    plt.close(fig)

    prob_diff = [[] for _ in range(n_interval)]

    model = network(h_dim, n_interval)
    model.cuda()
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    # begin training the model
    for i in range(iterations):
        optimizer.zero_grad()
        true_samples = inf_train_gen(data_type, batch_size=batch_size)
        idx = np.random.randint(low=0, high=n_interval, size=batch_size)
        left = al[idx]
        right = ar[idx]
        noise_samples = noise_injection(true_samples, left, right, baseline=baseline)
        noise_samples = torch.tensor(noise_samples, dtype=torch.float).cuda()
        idx = torch.tensor(idx, dtype=torch.long).cuda()
        logits = model(noise_samples)
        loss = loss_f(logits, idx).mean()
        loss.backward()
        optimizer.step()
        
        if i % print_iter == 0:
            _, pred = torch.max(logits.data, -1)
            acc = (pred == idx).float().mean().item()
            print("Iter {} time {:.2f} loss {:.3f} correct {:.3f}"\
                  .format(i, time.time() - start_time, loss.item(), acc))
        
        if i % plot_iter == 0 or i % moniter_iter == 0:
            # plot energy function
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            x_in, y_in = np.meshgrid(bin_centers, bin_centers)
            x_in = np.reshape(x_in, (-1, 1))
            y_in = np.reshape(y_in, (-1, 1))
            with torch.no_grad():
                test_data = torch.tensor(np.concatenate([x_in, y_in], axis=-1), dtype=torch.float).cuda()
                test_logits = model(test_data).detach()
                test_logits_numpy = test_logits.cpu().numpy()
                max_logits, _ = torch.max(test_logits, dim=0, keepdim=True)
                test_logits = test_logits - max_logits
                test_logits = torch.exp(test_logits) 
                test_logits = test_logits / torch.sum(test_logits, dim=0, keepdim=True)
                test_logits = test_logits.detach().cpu().numpy()
            
            if i % plot_iter == 0: 
                fig, axes = plt.subplots(nrow, n_interval // nrow)    
                for j in range(n_interval):
                    ii = j // (n_interval // nrow)
                    jj = j % (n_interval // nrow)
                    tmp = np.reshape(test_logits[:, j], (grid, grid))
                    tmp = np.flip(tmp, axis=0)
                    axes[ii, jj].matshow(tmp)
                    axes[ii, jj].tick_params(left = False, right = False, labelleft = False, labelbottom = False, labeltop = False, bottom = False, top=False)
                fig.savefig(os.path.join(img_dir, 'pred_dist_{}.png'.format(i)))
                plt.close(fig)
                
                fig, axes = plt.subplots(nrow, n_interval // nrow)    
                for j in range(n_interval):
                    ii = j // (n_interval // nrow)
                    jj = j % (n_interval // nrow)
                    tmp = np.reshape(test_logits_numpy[:, j], (grid, grid))
                    tmp = np.flip(tmp, axis=0)
                    axes[ii, jj].matshow(tmp)
                    axes[ii, jj].tick_params(left = False, right = False, labelleft = False, labelbottom = False, labeltop = False, bottom = False, top=False)
                fig.savefig(os.path.join(img_dir, 'energy_{}.png'.format(i)))
                plt.close(fig)

            if i % moniter_iter == 0:
                fig, axes = plt.subplots(nrow, n_interval // nrow)    
                for j in range(n_interval):
                    ii = j // (n_interval // nrow)
                    jj = j % (n_interval // nrow)
                    tmp = np.reshape(test_logits[:, j], (grid, grid))
                    tmp = np.flip(tmp, axis=0)
                    tmp_mse = np.sum((tmp - true_dist[j]) ** 2)
                    prob_diff[j].append(tmp_mse)
                    axes[ii, jj].plot(np.arange(len(prob_diff[j])), prob_diff[j])
                fig.savefig(os.path.join(img_dir, 'mse.png'))
                plt.close(fig)
                np.save(os.path.join(img_dir, 'mse.npy'), np.array(prob_diff))

                
        
        if (not i == 0) and (i % ckpt_iter == 0 or i == iterations - 1):
            print('Saving checkpoint')
            torch.save(model.state_dict(), os.path.join(ckpt_dir, '{}.pth.tar'.format(i)))

if __name__ == '__main__':
    train()