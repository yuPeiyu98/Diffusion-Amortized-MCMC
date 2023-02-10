import cv2
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(
        self, 
        **kwargs
    ):
        super(CIFAR10Dataset, self).__init__()

    def __getitem__(index: int) → Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        return img, index

###################################################################
#################### low-res datasets (32x32) #####################
###################################################################

class CIFAR10Dataset(Dataset):
    def __init__(
        self, 
        config, 
        data_split=0, 
        use_flip=True
    ):
        super(CIFAR10Dataset, self).__init__()        

        self.data_split = data_split
        self.use_flip = use_flip
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), 
                (0.5, 0.5, 0.5)
            )
        ])        
        
        self.root_dir = config.ROOT_DIR
        self.meta = self._collect_meta()

        self.im_res = [16, 8]

    def _collect_meta(self):
        """ 
        Returns a dictionary with image data and their labels
        as its values.

        Return: {
            'img': ndarray, (N, 3072),
            'lbl': ndarray, (N,)
        }
        """
        import pickle

        data_dir = self.root_dir

        meta = {'img': None, 'lbl': None}
        if self.data_split == 0:
            # training split
            for i in range(1, 6): # hard-coded for cifar-10 dataset
                with open(
                    osp.join(data_dir, 'data_batch_{}'.format(i)), 
                    'rb'
                ) as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                    if not isinstance(meta['img'], np.ndarray):
                        meta['img'] = data_dict[b'data']
                        meta['lbl'] = np.array(data_dict[b'labels'])
                    else:
                        meta['img'] = np.vstack([
                            meta['img'], 
                            data_dict[b'data']
                        ])
                        meta['lbl'] = np.hstack([
                            meta['lbl'], 
                            np.array(data_dict[b'labels'])
                        ])
        else:
            # testing / validation split
            with open(
                osp.join(data_dir, 'test_batch'), 
                'rb'
            ) as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                
                meta['img'] = data_dict[b'data']
                meta['lbl'] = np.array(data_dict[b'labels'])                
        
        print(
            '{} meta: img data {}, labels {}'.format(
                'training' if self.data_split == 0 else 'testing',
                meta['img'].shape, 
                meta['lbl'].shape
            )
        )
        return meta

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
        data = data.reshape(3, 32, 32).transpose(1, 2, 0)

        return self.transform(data), label

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

##### + under development
class MNISTDataset(Dataset):
    def __init__(
        self, 
        config, 
        data_split=0, 
        use_flip=True
    ):
        super(MNISTDataset, self).__init__()        

        self.data_split = data_split
        self.use_flip = use_flip
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), 
                (0.5, 0.5, 0.5)
            )
        ])        
        
        self.root_dir = config.ROOT_DIR
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
        import pickle

        data_dir = self.root_dir

        meta = {'img': None, 'lbl': None}
        for i in range(1, 6): # hard-coded for cifar-10 dataset
            with open(
                osp.join(data_dir, 'data_batch_{}'.format(i)), 
                'rb'
            ) as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                if meta['img'] == None:
                    meta['img'] = data_dict[b'data']
                    meta['lbl'] = np.array(data_dict[b'labels'])
                else:
                    meta['img'] = np.vstack([
                        meta['img'], 
                        data_dict[b'data']
                    ])
                    meta['lbl'] = np.hstack([
                        meta['lbl'], 
                        np.array(data_dict[b'labels'])
                    ])
        
        print(
            'Meta: img data {}, labels {}'.format(
                meta['img'].shape, meta['lbl'].shape
            )
        )
        return meta

    def __len__(self):
        return len(self.meta)

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
        data = data.reshape(3, 32, 32).transpose(1, 2, 0)            
        # pre-process the raw image
        data = self.transform(data)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
        return data, label

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item
