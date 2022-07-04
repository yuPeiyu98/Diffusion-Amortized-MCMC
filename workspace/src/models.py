import numpy as np
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .networks import UNet

###################################################################
####################### BASE MODEL & UTILS ########################
###################################################################

class BaseModel(nn.Module):

    def __init__(
        self, 
        name, 
        config
    ):
        super(BaseModel, self).__init__()

        self.model_name = name
        self.config = config        

        self.BASE_PATH = config.PATH
        self.iteration = 0

    def save(self):
        print('\nsaving {}...\n'.format(self.model_name))
        for __, module in enumerate(self.modules()):
            if hasattr(module, 'module_name') and \
                (module.module_name != 'Conv1x1' and \
                 module.module_name != 'Linear' and \
                 'in' not in module.module_name):
                module_path = osp.join(
                    self.BASE_PATH, 
                    '{}_{}.pth'.format(
                        self.model_name,
                        module.module_name
                    )
                )                

                # copy
                module_path_iter = osp.join(
                    self.BASE_PATH, 
                    '{}_{}_{}.pth'.format(
                        self.model_name,
                        module.module_name,
                        self.iteration
                    )
                )

                torch.save({
                    'iteration': self.iteration,
                    module.module_name: module.state_dict()
                }, module_path)

                torch.save({
                    'iteration': self.iteration,
                    module.module_name: module.state_dict()
                }, module_path_iter)

    def load(self):
        """ Retrieve saved modules """
        for __, module in enumerate(self.modules()):
            if hasattr(module, 'module_name'):
                module_path = osp.join(
                    self.BASE_PATH, 
                    '{}_{}.pth'.format(
                        self.model_name,
                        module.module_name
                    )
                )
                if os.path.exists(module_path):
                    print(
                        'Loading {} {}...'.format(
                            self.model_name, 
                            module.module_name
                        )
                    )

                    if torch.cuda.is_available():
                        meta = torch.load(module_path)
                    else:
                        meta = torch.load(
                            module_path, 
                            map_location=\
                                lambda storage, loc: storage
                        )

                    module.load_state_dict(meta[module.module_name])
                    self.iteration = meta['iteration']            

###################################################################
####################### U-Net Wake-Sleep ABP ######################
###################################################################

class ABPModel(BaseModel):

    def __init__(
        self,         
        config, 
        name='ABPModel',
        input_dim=None,
        latent_dim=None,
        use_scaled_output=True        
    ):
        super(ABPModel, self).__init__(name, config)
        self.im_size = int(config.IM_SIZE)

        # configuration
        self.config = config

        # langevin steps
        self.mcmc_steps = int(config.MCMC_STEPS)
        self.delta = float(config.DELTA)

        # network configuration
        self.use_var_head = bool(config.USE_VAR_HEAD)
        self.use_scaled_output = use_scaled_output

        self.input_dim = int(config.INPUT_DIM) \
                            if input_dim == None \
                            else input_dim
        self.latent_dim = int(config.LAT_DIM) \
                            if latent_dim == None \
                            else latent_dim

        self.encoder = UNet(
            output_dim=self.latent_dim,
            module_name='EncoderUNet',
            block_chns=[
                self.input_dim, 128, 256, 128, 64],
            r=.01,
            use_var_head=bool(config.USE_VAR_HEAD),
            use_spc_norm=bool(config.USE_SPC_NORM),
            init_weights=bool(config.INIT_WEIGHTS)
        )
        self.decoder = UNet(
            output_dim=self.input_dim,
            module_name='DecoderUNet',
            block_chns=[
                self.latent_dim, 128, 256, 128, 64],
            r=.01,
            use_spc_norm=bool(config.USE_SPC_NORM),
            init_weights=bool(config.INIT_WEIGHTS)
        )

        # optims
        self.optimizer = optim.Adam(
            params=[
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()}
            ],
            lr=float(config.LR),
            betas=(float(config.BETA1), float(config.BETA2))
        )        

        # reconstruction lss.
        self.sigma = float(config.SIGMA)

        # fnet weight
        self.f_weight = float(config.F_WEIGHT)

    @staticmethod
    def _set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=False for all the networks to 
           avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks 
                                     require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad 

    def _langevin_posterior_sampler(self, x, z_hat):
        self._set_requires_grad(self.decoder, requires_grad=False)

        # langevin sampling
        for __ in range(self.mcmc_steps):
            z_hat = z_hat.requires_grad_(True)
            cond_nll = 0.5 * F.mse_loss(
                            self._decoding(z_hat), x, reduction='none'
                        ).div(self.sigma ** 2)
            nll = cond_nll.sum() + .5 * z_hat.square().sum()
            
            d_z = torch.autograd.grad(nll, z_hat)[0]
            z_hat = z_hat - 0.5 * (self.delta ** 2) * d_z \
                          + self.delta * torch.randn_like(z_hat)
            z_hat = z_hat.detach()
        
        self._set_requires_grad(self.decoder, requires_grad=True)        

        return z_hat

    def _encoding(self, x):
        if self.use_var_head:
            z_mean, z_var = self.encoder(x)
            z = z_mean + torch.randn_like(z_mean) * (0.5 * z_var).exp()
        else:
            z = self.encoder(x)

        return z

    def _decoding(self, z):
        f = self.decoder(z)

        if self.use_scaled_output:
            return torch.tanh(f)

        return f

    def sleep_forward(self, x):
        z_prior = torch.randn(
                    x.size(0), self.latent_dim, self.im_size, self.im_size,
                    device=x.device
                )
        with torch.no_grad():
            x_prior = self._decoding(z_prior)
        z_prior_hat = self._encoding(x_prior)        
        return x_prior, z_prior_hat, z_prior

    def wake_forward(self, x):
        with torch.no_grad():
            z_hat = self._encoding(x)
        z = self._langevin_posterior_sampler(x, z_hat)
        return self._decoding(z), z_hat, z

    def update_model(self, x, x_rec, z, z_inf):
        ### calculate loss
        # + reconstruction loss
        l_rc = 0.5 * F.mse_loss(
                x_rec, x, reduction='none'
            ).div(self.sigma ** 2).sum(dim=[1, 2, 3]).mean()

        # + initializer loss
        l_f = F.mse_loss(z_inf, z)

        loss = l_rc + l_f * self.f_weight

        ### update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return l_rc.item(), l_f.item()

    def learn(self, x):
        self.iteration += 1

        x_prior, z_prior_hat, z_prior = self.sleep_forward(x)
        x_recon, z_hat, z = self.wake_forward(x)

        l_rc, l_f = self.update_model(x, x_recon, z_prior, z_prior_hat)

        logs = [
            ("l_rc", l_rc),
            ("l_f", l_f)
        ]

        return logs
