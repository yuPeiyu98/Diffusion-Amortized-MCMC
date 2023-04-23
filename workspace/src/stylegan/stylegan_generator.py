# python 3.7
"""Contains the generator class of StyleGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import os

import torch
import torch.nn as nn

from .stylegan_generator_network import StyleGANGeneratorNet

__all__ = ['StyleGANGenerator']


class StyleGANGenerator(nn.Module):
  """Defines the generator class of StyleGAN.

  Different from conventional GAN, StyleGAN introduces a disentangled latent
  space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
  the disentangled latent code, w, is fed into each convolutional layer to
  modulate the `style` of the synthesis through AdaIN (Adaptive Instance
  Normalization) layer. Normally, the w's fed into all layers are the same. But,
  they can actually be different to make different layers get different styles.
  Accordingly, an extended space (i.e. W+ space) is used to gather all w's
  together. Taking the official StyleGAN model trained on FF-HQ dataset as an
  instance, there are
  (1) Z space, with dimension (512,)
  (2) W space, with dimension (512,)
  (3) W+ space, with dimension (18, 512)
  """

  def __init__(self, weight_path, resolution=256):
    super().__init__()
    self.resolution = 256
    self.z_space_dim = getattr(self, 'z_space_dim', 512)
    self.w_space_dim = getattr(self, 'w_space_dim', 512)
    self.num_mapping_layers = getattr(self, 'num_mapping_layers', 8)
    self.repeat_w = getattr(self, 'repeat_w', False)
    self.image_channels = getattr(self, 'image_channels', 3)
    self.final_tanh = getattr(self, 'final_tanh', True)
    self.label_size = getattr(self, 'label_size', 0)
    self.fused_scale = getattr(self, 'fused_scale', 'auto')
    self.truncation_psi = 0.7
    self.truncation_layers = 8
    self.randomize_noise = False
    self.fmaps_base = getattr(self, 'fmaps_base', 16 << 10)
    self.fmaps_max = getattr(self, 'fmaps_max', 512)
    self.net = StyleGANGeneratorNet(
        resolution=self.resolution,
        z_space_dim=self.z_space_dim,
        w_space_dim=self.w_space_dim,
        num_mapping_layers=self.num_mapping_layers,
        repeat_w=self.repeat_w,
        image_channels=self.image_channels,
        final_tanh=self.final_tanh,
        label_size=self.label_size,
        fused_scale=self.fused_scale,
        truncation_psi=self.truncation_psi,
        truncation_layers=self.truncation_layers,
        randomize_noise=self.randomize_noise,
        fmaps_base=self.fmaps_base,
        fmaps_max=self.fmaps_max)
    self.num_layers = self.net.num_layers
    self.model_specific_vars = ['truncation.truncation']

    self.weight_path = weight_path

    if not os.path.isfile(self.weight_path):
      raise IOError('No pre-trained weights found for generator model!')
    state_dict = torch.load(self.weight_path)
    for var_name in self.model_specific_vars:
      state_dict[var_name] = self.net.state_dict()[var_name]
    self.net.load_state_dict(state_dict)
    self.net.eval()

  def forward(self, z, labels=None):
    """Synthesizes images with given latent codes.

    One can choose whether to generate the layer-wise style codes.

    Args:
      latent_codes: Input latent codes for image synthesis.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)
      labels: Additional labels for conditional generation.
      generate_style: Whether to generate the layer-wise style codes. (default:
        False)
      generate_image: Whether to generate the final image synthesis. (default:
        True)

    Returns:
      A dictionary whose values are raw outputs from the generator.
    """
    
    ls = None if labels is None else labels.float()

    # # Generate from Z space.
    # if z.ndim != 2 or z.shape[1] != self.z_space_dim:
    #   raise ValueError(f'Latent codes should be with shape [batch_size, '
    #                    f'latent_space_dim], where `latent_space_dim` equals '
    #                    f'to {self.z_space_dim}!\n'
    #                    f'But {z.shape} is received!')
    # ws = self.net.mapping(z, ls)
    # wps = self.net.truncation(ws)

    b = z.size(0)
    z = z.view(b, *[self.num_layers, self.w_space_dim])
    x = self.net.synthesis(z)

    return x