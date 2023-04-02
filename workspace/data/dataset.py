import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def adapt_labels(true_labels, label):
    """Adapt labels to anomaly detection context
    Args :
            true_labels (list): list of ints
            label (int): label which is considered anomalous
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    if label == 0:
        (true_labels[true_labels == label], true_labels[true_labels != label]) = (0, 1)
        true_labels = [1] * true_labels.shape[0] - true_labels
    else:
        (true_labels[true_labels != label], true_labels[true_labels == label]) = (0, 1)

    return true_labels

class MNIST(Dataset):
    def __init__(
        self, root, split, label, transform
    ):
        super().__init__()        
        self.split = split
        self.held_label = label
        self.transform = transform
        
        self.root_dir = root
        self.meta = self._collect_meta()

    def _collect_meta(self):
        """ 
        Returns a dictionary with image data and their labels
        as its values.

        Return: {
            'img': ndarray, (N, 3072),
            'lbl': ndarray, (N,)
        }
        """
        data = dict(np.load(osp.join(self.root_dir, 'mnist.npz')))
        dataset = {}

        full_x_data = np.concatenate([data['x_train'], data['x_test'], data['x_valid']], axis=0)
        full_y_data = np.concatenate([data['y_train'], data['y_test'], data['y_valid']], axis=0)

        normal_x_data = full_x_data[full_y_data != self.held_label]
        normal_y_data = full_y_data[full_y_data != self.held_label]

        RANDOM_SEED = 42
        RNG = np.random.RandomState(42)
        inds = RNG.permutation(normal_x_data.shape[0])
        normal_x_data = normal_x_data[inds]
        normal_y_data = normal_y_data[inds]

        index = int(normal_x_data.shape[0]*0.8)

        training_x_data = normal_x_data[:index]
        training_y_data = normal_y_data[:index]

        testing_x_data = np.concatenate([normal_x_data[index:], full_x_data[full_y_data == self.held_label]], axis=0)
        testing_y_data = np.concatenate([normal_y_data[index:], full_y_data[full_y_data == self.held_label]], axis=0)

        inds = RNG.permutation(testing_x_data.shape[0])
        testing_x_data = testing_x_data[inds]
        testing_y_data = testing_y_data[inds]

        if self.split == 'train':
            dataset['img'] = training_x_data
            dataset['lbl'] = adapt_labels(training_y_data, self.held_label)
        else:
            dataset['img'] = testing_x_data
            dataset['lbl'] = adapt_labels(testing_y_data, self.held_label)
        return dataset

    def __len__(self):
        return len(self.meta['img'])

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item    

    def load_item(self, index):
        data = self.meta['img'][index]        
        label = self.meta['lbl'][index]
        
        # reshape raw data to HxWxC format
        data = data.reshape(1, 28, 28).transpose(1, 2, 0)            
        # pre-process the raw image
        data = self.transform(data)

        return data, label
