# python 3.7
"""Contains the encoder class of StyleGAN inversion.
"""

import numpy as np

import os
import torch
import torch.nn as nn

from .stylegan_encoder_network import StyleGANEncoderNet

__all__ = ['StyleGANEncoder']


class StyleGANEncoder(nn.Module):
  """Defines the encoder class of StyleGAN inversion."""

  def __init__(self, weight_path, load=True, resolution=256):
    super().__init__()
    self.resolution = resolution

    self.w_space_dim = getattr(self, 'w_space_dim', 512)
    self.image_channels = getattr(self, 'image_channels', 3)
    self.encoder_channels_base = getattr(self, 'encoder_channels_base', 64)
    self.encoder_channels_max = getattr(self, 'encoder_channels_max', 1024)
    self.use_wscale = getattr(self, 'use_wscale', False)
    self.use_bn = getattr(self, 'use_bn', True)
    self.net = StyleGANEncoderNet(
        resolution=self.resolution,
        w_space_dim=self.w_space_dim,
        image_channels=self.image_channels,
        encoder_channels_base=self.encoder_channels_base,
        encoder_channels_max=self.encoder_channels_max,
        use_wscale=self.use_wscale,
        use_bn=self.use_bn)
    self.num_layers = self.net.num_layers
    self.encode_dim = [self.num_layers, self.w_space_dim]

    self.weight_path = weight_path
    if load:
        if not os.path.isfile(self.weight_path):
          raise IOError('No pre-trained weights found for Encoder model!')
        state_dict = torch.load(self.weight_path)
        self.net.load_state_dict(state_dict)
        self.net.eval()

  def forward(self, x):
    codes = self.net(x)
    assert codes.shape == (x.shape[0], np.prod(self.encode_dim))
    return codes