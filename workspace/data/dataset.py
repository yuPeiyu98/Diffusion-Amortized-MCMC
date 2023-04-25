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


import io
import os.path
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class LSUNClass(VisionDataset):
    def __init__(
        self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ) -> None:
        import lmdb

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        ##### Crop & resize to 256
        img = np.asarray(img)
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = Image.fromarray(img, 'RGB')
        img = img.resize((256, 256), Image.ANTIALIAS)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length


class LSUN(VisionDataset):
    """`LSUN <https://www.yf.io/p/lsun>`_ dataset.
    You will need to install the ``lmdb`` package to use this dataset: run
    ``pip install lmdb``
    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        classes: Union[str, List[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(root=os.path.join(root, f"{c}_lmdb"), transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
        categories = [
            "bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "tower",
        ]
        dset_opts = ["train", "val", "test"]

        try:
            classes = cast(str, classes)
            verify_str_arg(classes, "classes", dset_opts)
            if classes == "test":
                classes = [classes]
            else:
                classes = [c + "_" + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = "Expected type str or Iterable for argument classes, but got type {}."
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr_type = "Expected type str for elements in argument classes, but got type {}."
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split("_")
                category, dset_opt = "_".join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class", iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)

########################################################################
############################## CIFAR10 #################################
########################################################################
from torchvision.datasets import CIFAR10

class CIFAR10(VisionDataset):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, index

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

########################################################################
############################# MNIST ####################################
########################################################################

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
        self, root, split, label, transform, load=False
    ):
        super().__init__()        
        self.split = split
        self.held_label = label
        self.transform = transform
        self.load = load
        
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

        if self.load:
            dataset = np.load(
                osp.join(self.root_dir, 'heldout_{}_{}.npy'.format(self.held_label, self.split)),
                allow_pickle=True
            ).item()

        else:
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

            np.save(osp.join(self.root_dir, 'heldout_{}_{}.npy'.format(self.held_label, self.split)), dataset)
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
