import argparse
import cv2
import numpy as np
import os
import os.path as osp
import random
import torch
import torch.nn as nn

from data.dataset import *

from PIL import Image
from shutil import copyfile
from src.utils import Config, Progbar, create_dir
from src.utils import stitch_images, imsave

from src.models import ABPModel
from torch.utils.data import DataLoader

import torch.nn.functional as F

###################################################################
######################## helper functions #########################
###################################################################

def log(logs, log_file):
    with open(log_file, 'a') as f:
        f.write(
            '%s\n' % ' '.join([
                str(item[1]) for item in logs
            ])
        )

def cuda(args, device):
    return list(item.to(device) for item in args)

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def normalize(img):
    # [0, 255] => [0, 1]
    if img.max() - img.min() < 1e-5:
        if img.max() < 1e-5:
            img = torch.zeros(img.shape)
        else:
            img = torch.ones(img.shape)
    else:
        img = (img - img.min()) / (img.max() - img.min())
    return img

def save_gen(img, n, gen_path, model_name, token):
    for i, im in enumerate(img):
        im = (im.permute(1, 2, 0) + 1) / 2 * 255.0
        im = Image.fromarray(
            im.cpu().numpy().astype(np.uint8).squeeze()
        )

        path = osp.join(gen_path, model_name, token)
        name = osp.join(
            path, str(n + i).zfill(5) + ".png"
        )
        create_dir(path)            
        im.save(name)    

def viz(img, img_hat, iteration, samples_path, model_name):
    image_per_row = 2    

    images = stitch_images(
        postprocess((img + 1) / 2.),
        postprocess((img_hat + 1) / 2.),            
        img_per_row = image_per_row
    )

    path = osp.join(samples_path, model_name)
    name = osp.join(path, str(iteration).zfill(5) + ".png")
    create_dir(path)
    print('\nsaving sample ' + name)
    images.save(name)

###################################################################
###################### calculate likelihoods ######################
###################################################################

def log_gaussian(x, mean=None, log_var=None):
    import math

    b = x.size(0)

    if mean is None:
        mean = torch.zeros_like(x)
    if log_var is None:
        log_var = torch.zeros_like(x)
    
    log_p = - (x - mean) * (x - mean) / (2 * torch.exp(log_var)) \
            - 0.5 * log_var - 0.5 * math.log(math.pi * 2)
    
    return log_p.view(b, -1).sum(dim=-1)

def calculate_likelihood(model, data, sample_num=500):        
    x = data
    data_mask = torch.ones_like(data)

    w_pool = []
    for __ in range(sample_num):
        mu, logvar = model._encoding(x)
        z = model._reparameterize(mu, logvar, sample=True)
        x_hat = model._decoding(z)

        log_x_z = log_gaussian(x_hat, mean=x)
        log_z = log_gaussian(z)
        log_z_x = log_gaussian(z, mean=mu, log_var=logvar)
        # print(
        #     "[x|z]: {}, [z]: {}, [z|x]: {}".format(
        #         log_x_z.mean().item(),
        #         log_z.mean().item(),
        #         log_z_x.mean().item()
        #     )
        # )

        w = log_z + log_x_z - log_z_x

        w_pool.append(w)

    avg_exp_w = torch.cat(w_pool, dim=0).exp().mean()

    ll = avg_exp_w.log().item() if avg_exp_w > 0.0 else 0.0

    return ll

###################################################################
####################### evaluator scripts #########################
###################################################################

##### + single UVAE

def test_ABP(model, config, train_dataset, val_dataset):
    token = 'viz'
    samples_path = os.path.join(
        config.PATH, 'samples_test_{}'.format(token)
    )
    gen_path = os.path.join(
        config.PATH, 'generated_test_{}'.format(token)
    )
    log_file = osp.join(
        config.PATH, 'log_' + model.model_name + '_test_{}.dat'.format(token)
    )

    def test_mse():
        test_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=2,
            drop_last=False,
            shuffle=False
        )

        total = len(val_dataset)

        if total == 0:
            print('No data was provided! ' + \
                  'Check value in the configuration file.')
            return

        progbar = Progbar(
            total, 
            width=20, 
            stateful_metrics=['epoch', 'iter']
        )

        epoch = 0

        cum_mse = 0
        cnt_mse = 0

        for iteration, items in enumerate(test_loader):            
            model.eval()

            img, __ = cuda(items, config.DEVICE)

            # forward
            img_hat, z_hat, z = model.wake_forward(img)

            with torch.no_grad():
                mse = F.mse_loss(img_hat, img)
            cum_mse += mse
            cnt_mse += 1

            logs = [
                ("epoch", epoch),
                ("iter", iteration),
                ("mse", mse)
            ]

            progbar.add(len(img), values=logs)            

            # log model at checkpoints
            if bool(config.LOG_INTERVAL) and \
                iteration % int(config.LOG_INTERVAL) == 0:
                log(logs, log_file)

            # sample model at checkpoints
            if bool(config.SAMPLE_INTERVAL) and \
                iteration % int(config.SAMPLE_INTERVAL) == 0:
                viz(
                    img, img_hat, iteration, 
                    samples_path, model.model_name
                )
                
        print('\nMSE: {}'.format(cum_mse / cnt_mse))
        print('\nEnd MSE testing....')

    def test_fid():
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=2,
            drop_last=False,
            shuffle=False
        )

        total = len(train_dataset)

        if total == 0:
            print('No data was provided! ' + \
                  'Check value in the configuration file.')
            return

        progbar = Progbar(
            total, 
            width=20, 
            stateful_metrics=['epoch', 'iter']
        )

        epoch = 0        
        cnt = 0

        for iteration, items in enumerate(train_loader):
            model.eval()

            img, __ = cuda(items, config.DEVICE)

            # z = torch.randn(
            #         size=(len(img),
            #               int(config.LAT_DIM),
            #               int(config.INPUT_SIZE),
            #               int(config.INPUT_SIZE)),
            #         device=config.DEVICE
            #     )
            # with torch.no_grad():
            #     img_hat = model._decoding(z)
        
            logs = [
                ("epoch", epoch),
                ("iter", iteration)
            ]

            progbar.add(len(img), values=logs)

            # save generated sample
            save_gen(
                img, cnt, gen_path, model.model_name, 'gt'
            )
            # save_gen(
            #     img_hat, cnt, gen_path, model.model_name, 'gen'
            # )
            cnt += len(img) 
                
        print('\nEnd FID testing....')        

    test_mse()
    # test_fid()