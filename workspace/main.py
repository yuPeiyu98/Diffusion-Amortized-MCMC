import argparse
import cv2
import numpy as np
import os
import os.path as osp
import random
import torch
import torch.nn as nn

from data.dataset import *
from src.utils import Config
from src.models import ABPModel
from engine.trainer import run_ABP
# from engine.evaluator import test_ABP
from shutil import copyfile

###################################################################
#################### configuration & training #####################
###################################################################

def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, 
        reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', 
        '--checkpoints', 
        type=str, 
        default='./checkpoints', 
        help='model checkpoints path (default: ./checkpoints)')        

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)    

    return config

def get_dataset(config):
    # dataset configuration
    if config.DATA.lower() == "mnist":
        train_dataset = MNISTDataset(
            config, 
            data_split=config.TRAIN_SPLIT,
            use_flip=True
        )
        val_dataset = MNISTDataset(
            config, 
            data_split=config.VAL_SPLIT,
            use_flip=False
        )            
    elif config.DATA.lower() == "cifar10":
        train_dataset = CIFAR10Dataset(
            config, 
            data_split=config.TRAIN_SPLIT, 
            use_flip=True
        )
        val_dataset = CIFAR10Dataset(
            config, 
            data_split=config.VAL_SPLIT,
            use_flip=False
        )            
    else:
        raise ValueError("Unknown dataset.")
    return train_dataset, val_dataset    

def main(mode=None):
    r"""start the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, 
        reads from config file if not specified
    """

    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = \
                    ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        # cudnn auto-tuner
        # torch.backends.cudnn.benchmark = True   
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 
    # (prevents deadlocks with pytorch dataloader)
    # cv2.setNumThreads(0)

    # initialize random seed
    SEED = int(config.SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # build the model and initialize
    model = eval('{}Model'.format(config.MODEL))(
        config
    ).to(config.DEVICE)
    model.load()

    # load dataset
    train_dataset, val_dataset = get_dataset(config)

    # training    
    config.print()

    if not bool(config.EVAL_ONLY):
        print('\nstart training...\n')
        if bool(config.DEBUG):
            with torch.autograd.detect_anomaly():
                eval('run_{}'.format(config.MODEL))(
                    model, config, train_dataset, val_dataset
                )    
        else:
            eval('run_{}'.format(config.MODEL))(
                model, config, train_dataset, val_dataset
            )

    # testing
    # eval('test_{}'.format(config.MODEL))(
    #     model, config, train_dataset, val_dataset
    # )    

if __name__ == "__main__":
    main()
