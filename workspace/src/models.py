import numpy as np
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial

from .networks import UNet

###################################################################
####################### BASE MODEL & UTILS ########################
###################################################################

##### + HMC 

def Leapfrog(x, energy, step_size, L=3):
    """ Leapfrog integrator using Euclidean-Gaussian kinematic energy

    Args:
        x        -- initial samples
        energy   -- potential energy term U()
        L        -- iterations for leapfrog integrator
        stp_size -- step size for leapfrog integrator 

    It returns the (possibly) updated sample and the acceptance rate 
    of this batch data
    """        
    # initialize the dynamics and the momentum
    p0, _x = torch.randn_like(x), x.clone().detach().requires_grad_(True)
    
    # first half-step update for the momentum and 
    # the full step update for the data
    p = p0 + 0.5 * step_size * torch.autograd.grad(energy(_x).sum(), _x)[0]
    _x = _x + step_size * p
    for __ in range(L):
        p = p + step_size * torch.autograd.grad(energy(_x).sum(), _x)[0]
        _x = _x + step_size * p
    # the last half-step update for the momentum    
    p = p + 0.5 * step_size * torch.autograd.grad(energy(_x).sum(), _x)[0]

    # Metropolis-Hastings Correction
    H0 = -energy(x) + 0.5 * torch.sum(p0.square().view(p0.size(0), -1), 1)
    H1 = -energy(_x) + 0.5 * torch.sum(p.square().view(p.view(0), -1), 1)    
    p_acc = torch.minimum(torch.ones_like(H0), torch.exp(H0 - H1))
    replace_idx = p_acc > torch.rand_like(p_acc)
    x[replace_idx] = _x[replace_idx].detach().clone()

    acc_rate = torch.mean(replace_idx.float()).item()

    return x, acc_rate

##### + Base Model

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
        self.sampler = config.MCMC_SAMPLER
        self.mcmc_steps = int(config.MCMC_STEPS)
        self.delta = float(config.DELTA)
        self.step_mul = float(config.RATIO)
        self.L = int(config.L)
        self.accept_th = float(config.ACCEPT_THRESHOLD)

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
            nll = self._neg_log_prob_joint(z_hat, x).sum()
            
            d_z = torch.autograd.grad(nll, z_hat)[0]
            z_hat = z_hat - 0.5 * (self.delta ** 2) * d_z \
                          + self.delta * torch.randn_like(z_hat)
            z_hat = z_hat.detach()
        
        self._set_requires_grad(self.decoder, requires_grad=True)

        return z_hat

    def _hmc_posterior_sampler(self, x, z_hat):
        self._set_requires_grad(self.decoder, requires_grad=False)

        # hmc sampling
        step_sz = self.delta
        for __ in range(self.mcmc_steps):
            z_hat = z_hat.requires_grad_(True)

            z_hat, acc_rate = Leapfrog(
                                x=z_hat, energy=partial(self._neg_log_prob_joint, x=x), 
                                step_size=step_sz, L=self.L
                              )
            if acc_rate > self.accept_th:
                step_sz *= self.step_mul
            else:
                step_sz /= self.step_mul
        
        self._set_requires_grad(self.decoder, requires_grad=True)        

        return z_hat        

    def _neg_log_prob_joint(self, z, x):
        log_x_z = .5 * F.mse_loss(
                        self._decoding(z), x, reduction='none'
                    ).div(self.sigma ** 2).sum(dim=[1, 2, 3])
        log_z = .5 * z.square().sum(dim=1)
        return log_x_z + log_z

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
            x_recon_hat = self._decoding(z_hat)

        if self.sampler == "LD":
            z = self._langevin_posterior_sampler(x, z_hat)
        elif self.sampler == "HMC":
            z = self._hmc_posterior_sampler(x, z_hat)

        return self._decoding(z), x_recon_hat, z_hat, z

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
        x_recon, x_recon_hat, z_hat, z = self.wake_forward(x)

        l_rc, l_f = self.update_model(x, x_recon, z_prior, z_prior_hat)
        with torch.no_grad():
            l_rc_hat = 0.5 * F.mse_loss(
                    x_recon_hat, x, reduction='none'
                ).div(self.sigma ** 2).sum(dim=[1, 2, 3]).mean().item()

        logs = [
            ("l_rc", l_rc),
            ("l_rc_hat", l_rc_hat),
            ("l_f", l_f)
        ]

        return logs
