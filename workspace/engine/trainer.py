import argparse
import cv2
import numpy as np
import os
import os.path as osp
import random
import torch
import torch.nn as nn

from src.utils import Progbar, create_dir
from src.utils import stitch_images, imsave
from torch.utils.data import DataLoader

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
    # [min, max] => [0, 1]
    if img.max() - img.min() < 1e-5:
        if img.max() < 1e-5:
            img = torch.zeros(img.shape)
        else:
            img = torch.ones(img.shape)
    else:
        img = (img - img.min()) / (img.max() - img.min())
    return img

###################################################################
######################## trainer scripts ##########################
###################################################################

##### + ABP learning 

def run_ABP(model, config, train_dataset, val_dataset):

    sample_iterator = val_dataset.create_iterator(
        int(config.SAMPLE_SIZE)
    )
    sample_store_path = os.path.join(config.PATH, 'samples')

    log_file = osp.join(
        config.PATH, 'log_' + model.model_name + '.dat'
    )

    def sample(epoch):
        # do not sample when validation set is empty
        if len(val_dataset) == 0:
            return

        model.eval()

        # visualization configs
        iteration = model.iteration
        image_per_row = 2
        if int(config.SAMPLE_SIZE) <= 6:
            image_per_row = 1
        
        # visualize wake/sleep phase samples
        items = next(sample_iterator)
        img, __ = cuda(items, config.DEVICE)
        
        x_prior, z_prior_hat, z_prior = model.sleep_forward()
        img_hat, z_hat, z = model.wake_forward(img)        

        images = stitch_images(
            postprocess((img + 1) / 2.),
            postprocess((img_hat + 1) / 2.),
            postprocess(normalize(z_hat)),
            postprocess(normalize(z)),
            postprocess((x_prior + 1) / 2.),
            postprocess(normalize(z_prior_hat)),
            postprocess(normalize(z_prior)),
            img_per_row=image_per_row
        )

        path = osp.join(sample_store_path, model.model_name)
        name = osp.join(
            path, "{:03d}_{:05d}.png".format(epoch, iteration)
        )
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def train():
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=int(config.BATCH_SIZE),
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float(config.MAX_ITERS))
        total = len(train_dataset)

        if total == 0:
            print(
                'No training data was provided! ' + \
                'Check value in the configuration file.'
            )
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: {}'.format(epoch))

            progbar = Progbar(
                total, 
                width=20, 
                stateful_metrics=['epoch', 'iter']
            )

            for items in train_loader:
                model.train()

                img, __ = cuda(items, config.DEVICE)

                # learn                
                logs = model.learn(img)

                iteration = model.iteration
                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(img), values=logs)

                # log model at checkpoints
                if bool(config.LOG_INTERVAL) and \
                    iteration % int(config.LOG_INTERVAL) == 0:
                    log(logs, log_file)

                # sample model at checkpoints
                if bool(config.SAMPLE_INTERVAL) and \
                    iteration % int(config.SAMPLE_INTERVAL) == 0:
                    sample(epoch)                

                # save model at checkpoints
                if bool(config.SAVE_INTERVAL) and \
                    iteration % int(config.SAVE_INTERVAL) == 0:
                    model.save()

        print('\nEnd training....')
    
    train()