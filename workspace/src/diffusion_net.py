##############################################################################
# Include all the network structure (posterior, generator, prior) for training
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import numpy as np
import math
from .diffusion_helper_func import *

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

########### Generator ###########

class _netG_cifar10(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=3, use_spc_norm=False):
        super().__init__()
        self.nz = nz
        f = nn.LeakyReLU(0.2)

        self.gen = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(nz, ngf*8, 8, 1, 0, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*2, nc, 3, 1, 1),
                use_spc_norm
            ),
            nn.Tanh()
        )    
    
    def forward(self, z):
        _z = z.reshape((len(z), self.nz, 1, 1))
        return self.gen(_z)

class _netG_svhn(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, use_spc_norm=False):
        super().__init__()
        self.nz = nz
        f = nn.LeakyReLU(0.2)

        self.gen = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1),
                use_spc_norm
            ),
            nn.Tanh()
        )    
    
    def forward(self, z):
        _z = z.reshape((len(z), self.nz, 1, 1))
        return self.gen(_z)

class _netG_celeba64(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3, use_spc_norm=False):
        super().__init__()
        self.nz = nz
        f = nn.LeakyReLU(0.2)

        self.gen = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            nn.Tanh()
        )    
    
    def forward(self, z):
        _z = z.reshape((len(z), self.nz, 1, 1))
        return self.gen(_z)

class _netG_celebaHQ(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=3, use_spc_norm=False):
        super().__init__()
        self.nz = nz
        f = nn.LeakyReLU(0.2)

        self.gen = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*4, ngf*4, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            nn.Tanh()
        )    
    
    def forward(self, z):
        _z = z.reshape((len(z), self.nz, 1, 1))
        return self.gen(_z)

class _netG_mnist(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=1, use_spc_norm=False):
        super().__init__()
        self.nz = nz
        f = nn.LeakyReLU(0.2)

        self.gen = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(nz, ngf*8, 7, 1, 0, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = True),
                use_spc_norm
            ),
            f,
            spectral_norm(
                nn.ConvTranspose2d(ngf*2, nc, 3, 1, 1),
                use_spc_norm
            ),
            nn.Tanh()
        )    
    
    def forward(self, z):
        _z = z.reshape((len(z), self.nz, 1, 1))
        return self.gen(_z)

############ Latent EBM ############

class _netE(nn.Module):
    def __init__(self, nz=128, ndf=200, nez=1, e_sn=False):
        super().__init__()
        apply_sn = sn if e_sn else lambda x: x
        f = nn.LeakyReLU(0.2)
        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(nz, ndf)),
            f,

            apply_sn(nn.Linear(ndf, ndf)),
            f,

            apply_sn(nn.Linear(ndf, nez))
        )

    def forward(self, z):
        return self.ebm(z).squeeze()

############ Encoder Network ############

class Encoder_cifar10(nn.Module):
    def __init__(self, nc=3, nemb=128, nif=64, use_norm=True, use_spc_norm=False):
        super().__init__()
        self.norm = nn.InstanceNorm2d if use_norm else nn.Identity

        self.nemb = nemb
        modules = nn.Sequential(
            spectral_norm(
                nn.Conv2d(nc, nif, 3, 1, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=True),
                use_spc_norm,
            ),
            self.norm(nif * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nemb, 4, 1, 0),
                use_spc_norm
            )
        )
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).reshape((len(x), self.nemb))

class Encoder_celeba64(nn.Module):
    def __init__(self, nc=3, nemb=128, nif=64, use_norm=True, use_spc_norm=False):
        super().__init__()
        self.norm = nn.InstanceNorm2d if use_norm else nn.Identity

        self.nemb = nemb
        modules = nn.Sequential(
            spectral_norm(
                nn.Conv2d(nc, nif, 3, 1, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=True),
                use_spc_norm,
            ),
            self.norm(nif * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nemb, 4, 1, 0),
                use_spc_norm
            )
        )
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).reshape((len(x), self.nemb))

class Encoder_celebaHQ(nn.Module):
    def __init__(self, nc=3, nemb=128, nif=64, use_norm=True, use_spc_norm=False):
        super().__init__()
        self.norm = nn.InstanceNorm2d if use_norm else nn.Identity

        self.nemb = nemb
        modules = nn.Sequential(
            spectral_norm(
                nn.Conv2d(nc, nif, 3, 1, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=True),
                use_spc_norm,
            ),
            self.norm(nif * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 4, nif * 4, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nemb, 4, 1, 0),
                use_spc_norm
            )
        )
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).reshape((len(x), self.nemb))

class Encoder_mnist(nn.Module):
    def __init__(self, nc=1, nemb=128, nif=64, use_norm=True, use_spc_norm=False):
        super().__init__()
        self.norm = nn.InstanceNorm2d if use_norm else nn.Identity

        self.nemb = nemb
        modules = nn.Sequential(
            spectral_norm(
                nn.Conv2d(nc, nif, 3, 1, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=True),
                use_spc_norm,
            ),
            self.norm(nif * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=True),
                use_spc_norm
            ),
            self.norm(nif * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nif * 8, nemb, 3, 1, 0),
                use_spc_norm
            )
        )
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).reshape((len(x), self.nemb))

############ Diffusion Network ############

class ConcatSquashLinearSkipCtx(nn.Module):
    def __init__(self, dim_in, dim_out, nxemb, ntemb, use_spc_norm=False):
        super(ConcatSquashLinearSkipCtx, self).__init__()
        self._layer = nn.Sequential(
            spectral_norm(
                nn.Linear(dim_in, dim_out), 
                use_spc_norm
            )
        )
        self._layer_ctx = nn.Sequential( 
            nn.SiLU(),
            spectral_norm(
                nn.Linear(ntemb + nxemb, dim_out),
                use_spc_norm
            ),
            nn.SiLU()
        )

        self._hyper_bias = spectral_norm(nn.Linear(dim_out, dim_out, bias=False), use_spc_norm)
        self._hyper_gate = spectral_norm(nn.Linear(dim_out, dim_out), use_spc_norm)
        self._skip = spectral_norm(nn.Linear(dim_in, dim_out), use_spc_norm)

    def forward(self, ctx, x):
        ctx = self._layer_ctx(ctx)

        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret + self._skip(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_time=1000.):
        super().__init__()
        self.dim = dim
        self.max_time = max_time

    def forward(self, x):
        x *= (1000. / self.max_time)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb 

class Diffusion_UnetA(nn.Module):
    def __init__(self, nz=128, nxemb=128, ntemb=128, residual=False, nf=4):
        super().__init__()
        self.act = F.leaky_relu # F.silu
        self.nz = nz
        self.nxemb = nxemb
        self.ntemb = ntemb 
        self.residual = residual
        
        sinu_pos_emb = SinusoidalPosEmb(ntemb, max_time=1.)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(ntemb, ntemb),
            nn.SiLU(),
            nn.Linear(ntemb, ntemb)
        )
        self.B = nn.Parameter(data=torch.randn(nz, nz // 2), requires_grad=True)
        
        self.in_layers = nn.ModuleList([
            ConcatSquashLinearSkipCtx(nz * 2, 32 * nf, nxemb, ntemb),
            ConcatSquashLinearSkipCtx(32 * nf, 64 * nf, nxemb, ntemb),
            ConcatSquashLinearSkipCtx(64 * nf, 64 * nf, nxemb, ntemb),      
        ])
        
        self.mid_layers = nn.ModuleList([
            ConcatSquashLinearSkipCtx(64 * nf, 64 * nf, nxemb, ntemb),
        ])

        self.out_layers = nn.ModuleList([
            ConcatSquashLinearSkipCtx(128 * nf, 64 * nf, nxemb, ntemb),
            ConcatSquashLinearSkipCtx(128 * nf, 32 * nf, nxemb, ntemb),
            ConcatSquashLinearSkipCtx(64 * nf, nz, nxemb, ntemb)
        ])

    def input_emb(self, x):
        return torch.cat([torch.sin(2 * np.pi * torch.matmul(x, self.B)), 
                          torch.cos(2 * np.pi * torch.matmul(x, self.B)), x], dim=1)

    def forward(self, z, logsnr, xemb):
        b = len(z)
        assert z.shape == (b, self.nz)
        assert logsnr.shape == (b,)
        assert (xemb is None and self.nxemb == 0) or xemb.shape == (b, self.nxemb)
        logsnr_input = (torch.arctan(torch.exp(-0.5 * torch.clamp(logsnr, min=-20., max=20.))) / (0.5 * np.pi))
        temb = self.time_mlp(logsnr_input)
        assert temb.shape == (b, self.ntemb)
        if xemb is None:
            total_emb = temb
        else:
            total_emb = torch.cat([temb, xemb], dim=1)

        hs = []
        out = self.input_emb(z)
        for i, layer in enumerate(self.in_layers):
            out = layer(ctx=total_emb, x=out)
            hs.append(out)
            out = self.act(out, negative_slope=0.01)
        out = self.mid_layers[0](ctx=total_emb, x=out)
        for i, layer in enumerate(self.mid_layers[1:]):
            out = self.act(out, negative_slope=0.01)
            out = layer(ctx=total_emb, x=out)
        for i, layer in enumerate(self.out_layers):
            out = torch.cat([out, hs.pop()], dim=1)
            out = self.act(out, negative_slope=0.01)
            out = layer(ctx=total_emb, x=out)
            
        assert out.shape == (b, self.nz)
        if self.residual:
            return z + out
        else:
            return out

############ Diffusion-Based Amortizer ############

class _netQ_U(nn.Module):
    def __init__(self, 
        nc=3, 
        nz=128, 
        nxemb=128, 
        ntemb=128,
        nf=4, 
        nif=64, 
        diffusion_residual=False,
        n_interval=20,
        logsnr_min=-20.,
        logsnr_max=20., 
        var_type='small', # try 'large', 'small'
        with_noise=False, 
        cond_w=0,
        net_arch='A',
        dataset='cifar10'
    ):

        super().__init__()
        print("Conditional model Q", with_noise)
        self.n_interval = n_interval
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.var_type = var_type
        self.nz = nz
        self.nxemb = nxemb
        self.with_noise = with_noise
        if dataset == 'cifar10' or dataset == 'svhn':
            self.encoder = Encoder_cifar10(nc=nc, nemb=nxemb, nif=nif)
        elif dataset == 'mnist':
            self.encoder = Encoder_mnist(nc=1, nemb=nxemb, nif=nif)
        elif dataset == 'celeba64':
            self.encoder = Encoder_celeba64(nc=nc, nemb=nxemb, nif=nif)
        else:
            self.encoder = Encoder_celebaHQ(nc=nc, nemb=nxemb, nif=nif)

        self.p = Diffusion_UnetA(nz=nz, nxemb=nxemb, ntemb=ntemb, residual=diffusion_residual, nf=nf)
    
        self.xemb = nn.Parameter(data=torch.randn(1, self.nxemb), requires_grad=True)
        self.prior_emb = nn.Sequential(
            nn.Linear(nz, 128),
            nn.LeakyReLU(),
            nn.Linear(128, nxemb)
        )

        self.cond_w = cond_w

    def forward(self, x=None, b=None, device=None, cond_w=-1):
        # give x infer z
        if x is not None:
            assert b is None and device is None
            b = len(x)
            xemb = self.encoder(x)
            device = x.device
        else:
            xemb = self.prior_emb(torch.randn(b, self.nz, device=device))

        zt = torch.randn(b, self.nz).to(device)
        
        for i in reversed(range(0, self.n_interval)):
            i_tensor = torch.ones(b, dtype=torch.float).to(device) * float(i)
            logsnr_t = logsnr_schedule_fn(i_tensor / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            logsnr_s = logsnr_schedule_fn(torch.clamp(i_tensor - 1.0, min=0.0) / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            eps_pred = self.p(z=zt, logsnr=logsnr_t, xemb=xemb)

            if x is not None and cond_w > 0:
                xemb_unc = self.prior_emb(torch.randn(b, self.nz, device=device))
                eps_pred_unc = self.p(z=zt, logsnr=logsnr_t, xemb=xemb_unc)
                eps_pred = (1 + cond_w) * eps_pred - cond_w * eps_pred_unc
            
            logsnr_t = logsnr_t.reshape((b, 1))
            logsnr_s = logsnr_s.reshape((b, 1))
            pred_z = pred_x_from_eps(z=zt, eps=eps_pred, logsnr=logsnr_t)

            if i == 0:
                zt = pred_z
            else:
                z_s_dist = diffusion_reverse(x=pred_z, z_t=zt, logsnr_s=logsnr_s, logsnr_t=logsnr_t, pred_var_type=self.var_type)
                eps = torch.randn_like(zt)
                if self.with_noise:
                    zt = z_s_dist['mean'] + z_s_dist['std'] * eps
                else:
                    zt = z_s_dist['mean']

        return zt   
        
    def calculate_loss(self, x=None, z=None, mask=None):
        # given inferred x and z train diffusion model
        assert z is not None
        if x is not None: 
            xemb = self.encoder(x)
            if mask is not None:
                xemb = xemb * mask \
                     + self.prior_emb(torch.randn(len(x), self.nz, device=x.device)) * (1 - mask)
        else:
            assert mask is None
            xemb = self.prior_emb(torch.randn(len(z), self.nz, device=z.device))

        u = torch.rand(len(z)).to(z.device)
        logsnr = logsnr_schedule_fn(u, logsnr_max=self.logsnr_max, logsnr_min=self.logsnr_min)

        zt_dist = diffusion_forward(z, logsnr=logsnr.reshape(len(z), 1))
        eps = torch.randn_like(z)
        zt = zt_dist['mean'] + zt_dist['std'] * eps
        eps_pred = self.p(z=zt, logsnr=logsnr, xemb=xemb)
        assert eps.shape == eps_pred.shape == (len(z), self.nz)
        loss = 0.5 * torch.sum((eps - eps_pred) ** 2, dim=1)

        return loss