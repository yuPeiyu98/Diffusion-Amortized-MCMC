# python 3.7
"""Contains the encoder class of StyleGAN inversion.
"""

import numpy as np

import torch

from .stylegan_encoder_network import StyleGANEncoderNet

__all__ = ['StyleGANEncoder']


class StyleGANEncoder(nn.Module):
  """Defines the encoder class of StyleGAN inversion."""

  def __init__(self, resolution=256):
    super().__init__()
    self.resolution = resolution

    self.w_space_dim = getattr(self, 'w_space_dim', 512)
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

  def forward(self, x):
    codes = self.net(x).reshape(x.size(0), -1)
    assert codes.shape == (x.shape[0], np.prod(self.encode_dim))
    return codes