# python 3.7
"""Contains the VGG16 model for perceptual feature extraction."""

import os
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

_MEAN_STATS = (103.939, 116.779, 123.68)


class VGG16(nn.Sequential):
  """Defines the VGG16 structure as the perceptual network.

  This models takes `RGB` images with pixel range [-1, 1] and data format `NCHW`
  as raw inputs. This following operations will be performed to preprocess the
  inputs (as defined in `keras.applications.imagenet_utils.preprocess_input`):
  (1) Shift pixel range to [0, 255].
  (3) Change channel order to `BGR`.
  (4) Subtract the statistical mean.

  NOTE: The three fully connected layers on top of the model are dropped.
  """

  def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
    """Defines the network structure.

    Args:
      output_layer_idx: Index of layer whose output will be used as perceptual
        feature. (default: 23, which is the `block4_conv3` layer activated by
        `ReLU` function)
      min_val: Minimum value of the raw input. (default: -1.0)
      max_val: Maximum value of the raw input. (default: 1.0)
    """
    sequence = OrderedDict({
        'layer0': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        'layer1': nn.ReLU(inplace=True),
        'layer2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        'layer3': nn.ReLU(inplace=True),
        'layer4': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer5': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        'layer6': nn.ReLU(inplace=True),
        'layer7': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        'layer8': nn.ReLU(inplace=True),
        'layer9': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer10': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        'layer11': nn.ReLU(inplace=True),
        'layer12': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        'layer13': nn.ReLU(inplace=True),
        'layer14': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        'layer15': nn.ReLU(inplace=True),
        'layer16': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer17': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        'layer18': nn.ReLU(inplace=True),
        'layer19': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer20': nn.ReLU(inplace=True),
        'layer21': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer22': nn.ReLU(inplace=True),
        'layer23': nn.MaxPool2d(kernel_size=2, stride=2),
        'layer24': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer25': nn.ReLU(inplace=True),
        'layer26': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer27': nn.ReLU(inplace=True),
        'layer28': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer29': nn.ReLU(inplace=True),
        'layer30': nn.MaxPool2d(kernel_size=2, stride=2),
    })
    self.output_layer_idx = output_layer_idx
    self.min_val = min_val
    self.max_val = max_val
    self.mean = torch.tensor(_MEAN_STATS).float().view(1, 3, 1, 1)
    super().__init__(sequence)

  def forward(self, x):
    x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
    x = x[:, [2, 1, 0], :, :]
    x = x - self.mean.to(x.device)
    for i in range(self.output_layer_idx):
      x = self.__getattr__(f'layer{i}')(x)
    return x


class PerceptualModel(nn.Module):
  """Defines the perceptual model class."""

  def __init__(self, weight_path, output_layer_idx=23, min_val=-1.0, max_val=1.0):
    """Initializes."""
    self.output_layer_idx = output_layer_idx
    self.image_channels = 3
    self.min_val = min_val
    self.max_val = max_val
    self.net = VGG16(output_layer_idx=self.output_layer_idx,
                     min_val=self.min_val,
                     max_val=self.max_val)

    self.weight_path = weight_path

    if not os.path.isfile(self.weight_path):
      raise IOError('No pre-trained weights found for perceptual model!')
    self.net.load_state_dict(torch.load(self.weight_path))
    self.net.eval()

  def forward(self, x):
    """Extracts perceptual feature within mini-batch."""
    return self.net(x)