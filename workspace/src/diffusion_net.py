# ############################################################################
# Include all the network structure (posterior, generator, prior) for training
# ############################################################################
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

########### Generator ###################
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

############ Latent EBM ##################
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
        return self.ebm(z.squeeze()).squeeze()

############# Inference model #############
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

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        #self._layer.weight.data = 1e-4 * torch.randn_like(self._layer.weight.data)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        #self._hyper_bias.weight.data.zero_()
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        #self._hyper_gate.weight.data.zero_()

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret

class ConcatSquashLinearSkip(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinearSkip, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        #self._layer.weight.data = 1e-4 * torch.randn_like(self._layer.weight.data)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        #self._hyper_bias.weight.data.zero_()
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        #self._hyper_gate.weight.data.zero_()
        self._skip = nn.Linear(dim_in, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret + self._skip(x)

class ConcatSquashLinearSkipCtx(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx, use_spc_norm=False):
        super(ConcatSquashLinearSkipCtx, self).__init__()

        self._layer = spectral_norm(nn.Linear(dim_in, dim_out), use_spc_norm)
        self._layer_ctx = nn.Sequential( 
            nn.SiLU(),
            spectral_norm(
                nn.Linear(dim_ctx, dim_ctx // 2),
                use_spc_norm
            ),
            nn.SiLU()
        )

        #self._layer.weight.data = 1e-4 * torch.randn_like(self._layer.weight.data)
        self._hyper_bias = spectral_norm(nn.Linear(dim_ctx // 2, dim_out, bias=False), use_spc_norm)
        #self._hyper_bias.weight.data.zero_()
        self._hyper_gate = spectral_norm(nn.Linear(dim_ctx // 2, dim_out), use_spc_norm)
        #self._hyper_gate.weight.data.zero_()
        self._skip = spectral_norm(nn.Linear(dim_in, dim_out), use_spc_norm)

    def forward(self, ctx, x):
        ctx = self._layer_ctx(ctx)

        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret + self._skip(x)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads=1):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class ConcatSquashLinearSkipCtxAttn(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinearSkipCtxAttn, self).__init__()
        self.dim_out = dim_out

        self._layer_in = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU()
        )
        self._layer_ctx = nn.Sequential( 
            nn.SiLU(),
            nn.Linear(dim_ctx, dim_out),
            nn.SiLU()
        )
        
        self.x_lift = nn.Conv1d(1, 128 * 3, 1)
        self.c_lift = nn.Conv1d(1, 128 * 2, 1)
        self.attention = QKVAttention(n_heads=1)
        self.xc_sqz = nn.Conv1d(128, 1, 1)
        self.out = nn.Linear(dim_out, dim_out)

        self._skip = nn.Linear(dim_in, dim_out)

    def forward(self, ctx, x):
        b = x.size(0)

        x_l = self._layer_in(x)
        ctx_l = self._layer_ctx(ctx)

        # extend dim.
        x_l = x_l.reshape(b, 1, self.dim_out)
        ctx_l = ctx_l.reshape(b, 1, self.dim_out)

        # attn.
        qkv = self.x_lift(x_l)
        encoder_out = self.c_lift(ctx_l)
        h = self.attention(qkv, encoder_out)

        ret = self.out(self.xc_sqz(h).squeeze(1))

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

class Diffusion_net(nn.Module):
    def __init__(self, nz=128, nxemb=128, ntemb=128, residual=False):
        super().__init__()
        self.act = F.leaky_relu
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
        
        self.layers = nn.ModuleList([
            ConcatSquashLinear(nz, 128, nxemb + ntemb),
            ConcatSquashLinear(128, 256, nxemb + ntemb),
            ConcatSquashLinear(256, 512, nxemb + ntemb),
            ConcatSquashLinear(512, 256, nxemb + ntemb),
            ConcatSquashLinear(256, 128, nxemb + ntemb),
            ConcatSquashLinear(128, nz, nxemb + ntemb)
        ])
        #self.layers[-1]._layer.weight.data.zero_()
    
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

        out = z
        for i, layer in enumerate(self.layers):
            out = layer(ctx=total_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out, negative_slope=0.01)
        assert out.shape == (b, self.nz)
        if self.residual:
            return z + out
        else:
            return out

class Diffusion_Unet(nn.Module):
    def __init__(self, nz=128, nxemb=128, ntemb=128, residual=False):
        super().__init__()
        self.act = F.leaky_relu
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
        
        self.in_layers = nn.ModuleList([
            ConcatSquashLinearSkip(nz, 128, nxemb + ntemb),
            ConcatSquashLinearSkip(128, 256, nxemb + ntemb),
            ConcatSquashLinearSkip(256, 256, nxemb + ntemb)           
        ])
        #self.layers[-1]._layer.weight.data.zero_()

        self.mid_layer = ConcatSquashLinearSkip(256, 256, nxemb + ntemb) 
    
        self.out_layers = nn.ModuleList([
            ConcatSquashLinearSkip(512, 256, nxemb + ntemb),
            ConcatSquashLinearSkip(512, 128, nxemb + ntemb),
            ConcatSquashLinearSkip(256, nz, nxemb + ntemb)
        ])

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
        out = z
        for i, layer in enumerate(self.in_layers):
            out = layer(ctx=total_emb, x=out)
            hs.append(out)
            out = self.act(out, negative_slope=0.01)
        out = self.mid_layer(ctx=total_emb, x=out)
        for i, layer in enumerate(self.out_layers):
            out = torch.cat([out, hs.pop()], dim=1)
            out = self.act(out, negative_slope=0.01)
            out = layer(ctx=total_emb, x=out)
            
        assert out.shape == (b, self.nz)
        if self.residual:
            return z + out
        else:
            return out

class Diffusion_UnetA(nn.Module):
    def __init__(self, nz=128, nxemb=128, ntemb=128, residual=False):
        super().__init__()
        self.act = F.leaky_relu
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
            ConcatSquashLinearSkipCtx(nz * 2, 128, nxemb + ntemb),
            ConcatSquashLinearSkipCtx(128, 256, nxemb + ntemb),
            ConcatSquashLinearSkipCtx(256, 256, nxemb + ntemb)           
        ])
        #self.layers[-1]._layer.weight.data.zero_()

        self.mid_layer = ConcatSquashLinearSkipCtx(256, 256, nxemb + ntemb) 
    
        self.out_layers = nn.ModuleList([
            ConcatSquashLinearSkipCtx(512, 256, nxemb + ntemb),
            ConcatSquashLinearSkipCtx(512, 128, nxemb + ntemb),
            ConcatSquashLinearSkipCtx(256, nz, nxemb + ntemb)
        ])

    def input_emb(self, x):
        # x_1 = 2. * np.pi * x
        # x_7 = np.power(2, 7) * np.pi * x
        # x_8 = np.power(2, 8) * np.pi * x

        # x_proj = torch.cat([x_1, x_7, x_8], dim=1)

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
        out = self.mid_layer(ctx=total_emb, x=out)
        for i, layer in enumerate(self.out_layers):
            out = torch.cat([out, hs.pop()], dim=1)
            out = self.act(out, negative_slope=0.01)
            out = layer(ctx=total_emb, x=out)
            
        assert out.shape == (b, self.nz)
        if self.residual:
            return z + out
        else:
            return out

class _netQ(nn.Module):
    def __init__(self, 
        nc=3, 
        nz=128, 
        nxemb=128, 
        ntemb=128, 
        nif=64, 
        diffusion_residual=False,
        n_interval=20,
        logsnr_min=-20.,
        logsnr_max=20., 
        var_type='small', # try 'large', 'small'
        with_noise=False, 
        cond_w=0
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
        self.encoder = Encoder_cifar10(nc=nc, nemb=nxemb, nif=nif)
        self.p = Diffusion_net(nz=nz, nxemb=nxemb, ntemb=ntemb, residual=diffusion_residual)

        self.cond_w = cond_w

    def forward(self, x=None, b=None, device=None):
        # give x infer z
        if x is not None:
            assert b is None and device is None
            b = len(x)
            xemb = self.encoder(x)
            device = x.device
        else:
            xemb = torch.zeros(b, self.nxemb).to(device)
        zt = torch.randn(b, self.nz).to(device)
        #print('zt', zt.max(), zt.min())
        for i in reversed(range(0, self.n_interval)):
            i_tensor = torch.ones(b, dtype=torch.float).to(device) * float(i)
            logsnr_t = logsnr_schedule_fn(i_tensor / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            logsnr_s = logsnr_schedule_fn(torch.clamp(i_tensor - 1.0, min=0.0) / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            eps_pred = self.p(z=zt, logsnr=logsnr_t, xemb=xemb)

            if x is not None and self.cond_w > 0:
                eps_pred_unc = self.p(z=zt, logsnr=logsnr_t, xemb=torch.zeros(b, self.nxemb).to(device))
                eps_pred = (1 + self.cond_w) * eps_pred - self.cond_w * eps_pred_unc
            
            #print('eps', i, eps_pred.max(), eps_pred.min())
            logsnr_t = logsnr_t.reshape((b, 1))
            logsnr_s = logsnr_s.reshape((b, 1))
            pred_z = pred_x_from_eps(z=zt, eps=eps_pred, logsnr=logsnr_t)
            #print('pred_z', i, pred_z.max(), pred_z.min())
            #pred_z = torch.clamp(pred_z, min=-2.5, max=2.5)

            if i == 0:
                zt = pred_z
            else:
                z_s_dist = diffusion_reverse(x=pred_z, z_t=zt, logsnr_s=logsnr_s, logsnr_t=logsnr_t, pred_var_type=self.var_type)
                eps = torch.randn_like(zt)
                # if self.with_noise or x is None:
                if self.with_noise:
                    zt = z_s_dist['mean'] + z_s_dist['std'] * eps
                else:
                    zt = z_s_dist['mean']

        return zt   
        
    def calculate_loss(self, x=None, z=None, mask=None):
        # given inferred x and z train diffusion model
        #assert len(x) == len(z)
        assert z is not None
        if x is not None: 
            xemb = self.encoder(x)
            if mask is not None:
                xemb = xemb * mask
        else:
            assert mask is None
            xemb = torch.zeros(len(z), self.nxemb).to(z.device)
        u = torch.rand(len(z)).to(z.device)
        logsnr = logsnr_schedule_fn(u, logsnr_max=self.logsnr_max, logsnr_min=self.logsnr_min)

        zt_dist = diffusion_forward(z, logsnr=logsnr.reshape(len(z), 1))
        eps = torch.randn_like(z)
        zt = zt_dist['mean'] + zt_dist['std'] * eps
        eps_pred = self.p(z=zt, logsnr=logsnr, xemb=xemb)
        assert eps.shape == eps_pred.shape == (len(z), self.nz)
        loss = 0.5 * torch.sum((eps - eps_pred) ** 2, dim=1)
        return loss

class _netQ_U(nn.Module):
    def __init__(self, 
        nc=3, 
        nz=128, 
        nxemb=128, 
        ntemb=128, 
        nif=64, 
        diffusion_residual=False,
        n_interval=20,
        logsnr_min=-20.,
        logsnr_max=20., 
        var_type='small', # try 'large', 'small'
        with_noise=False, 
        cond_w=0,
        net_arch='A'
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
        self.encoder = Encoder_cifar10(nc=nc, nemb=nxemb, nif=nif)

        if net_arch == 'vanilla':
            self.p = Diffusion_Unet(nz=nz, nxemb=nxemb, ntemb=ntemb, residual=diffusion_residual)
        elif net_arch == 'A':
            self.p = Diffusion_UnetA(nz=nz, nxemb=nxemb, ntemb=ntemb, residual=diffusion_residual)
        elif net_arch == 'B':
            self.p = Diffusion_UnetB(nz=nz, nxemb=nxemb, ntemb=ntemb, residual=diffusion_residual)
        elif net_arch == 'C':
            self.p = Diffusion_UnetC(nz=nz, nxemb=nxemb, ntemb=ntemb, residual=diffusion_residual)

        self.xemb = nn.Parameter(data=torch.randn(1, self.nxemb), requires_grad=True)

        self.cond_w = cond_w

    def forward(self, x=None, b=None, device=None):
        # give x infer z
        if x is not None:
            assert b is None and device is None
            b = len(x)
            xemb = self.encoder(x)
            device = x.device
        else:
            # xemb = torch.zeros(b, self.nxemb).to(device)
            xemb = self.xemb.expand(b, -1)

        zt = torch.randn(b, self.nz).to(device)
        #print('zt', zt.max(), zt.min())
        for i in reversed(range(0, self.n_interval)):
            i_tensor = torch.ones(b, dtype=torch.float).to(device) * float(i)
            logsnr_t = logsnr_schedule_fn(i_tensor / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            logsnr_s = logsnr_schedule_fn(torch.clamp(i_tensor - 1.0, min=0.0) / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            eps_pred = self.p(z=zt, logsnr=logsnr_t, xemb=xemb)

            if x is not None and self.cond_w > 0:
                eps_pred_unc = self.p(z=zt, logsnr=logsnr_t, xemb=torch.zeros(b, self.nxemb).to(device))
                eps_pred = (1 + self.cond_w) * eps_pred - self.cond_w * eps_pred_unc
            
            #print('eps', i, eps_pred.max(), eps_pred.min())
            logsnr_t = logsnr_t.reshape((b, 1))
            logsnr_s = logsnr_s.reshape((b, 1))
            pred_z = pred_x_from_eps(z=zt, eps=eps_pred, logsnr=logsnr_t)
            #print('pred_z', i, pred_z.max(), pred_z.min())
            #pred_z = torch.clamp(pred_z, min=-2.5, max=2.5)

            if i == 0:
                zt = pred_z
            else:
                z_s_dist = diffusion_reverse(x=pred_z, z_t=zt, logsnr_s=logsnr_s, logsnr_t=logsnr_t, pred_var_type=self.var_type)
                eps = torch.randn_like(zt)
                # if self.with_noise or x is None:
                if self.with_noise:
                    zt = z_s_dist['mean'] + z_s_dist['std'] * eps
                else:
                    zt = z_s_dist['mean']

        return zt   
        
    def calculate_loss(self, x=None, z=None, mask=None):
        # given inferred x and z train diffusion model
        #assert len(x) == len(z)
        assert z is not None
        if x is not None: 
            xemb = self.encoder(x)
            if mask is not None:
                xemb = xemb * mask + self.xemb.expand(len(x), -1) * (1 - mask)
        else:
            assert mask is None
            # xemb = torch.zeros(len(z), self.nxemb).to(z.device)
            xemb = self.xemb.expand(len(x), -1)

        u = torch.rand(len(z)).to(z.device)
        logsnr = logsnr_schedule_fn(u, logsnr_max=self.logsnr_max, logsnr_min=self.logsnr_min)

        zt_dist = diffusion_forward(z, logsnr=logsnr.reshape(len(z), 1))
        eps = torch.randn_like(z)
        zt = zt_dist['mean'] + zt_dist['std'] * eps
        eps_pred = self.p(z=zt, logsnr=logsnr, xemb=xemb)
        assert eps.shape == eps_pred.shape == (len(z), self.nz)
        loss = 0.5 * torch.sum((eps - eps_pred) ** 2, dim=1)
        return loss

class _netQ_uncond(nn.Module):
    def __init__(self, 
        nc=3, 
        nz=128, 
        ntemb=128, 
        diffusion_residual=False,
        n_interval=20,
        logsnr_min=-20.,
        logsnr_max=20., 
        var_type='small', # try 'large', 'small'
        ):

        super().__init__()
        print("Unconditional prior model")
        self.n_interval = n_interval
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.var_type = var_type
        self.nz = nz
        self.p = Diffusion_net(nz=nz, nxemb=0, ntemb=ntemb, residual=diffusion_residual) # no x embedding

    def forward(self, x=None, b=None, device=None):
        # generate prior samples
        assert x is None and b is not None and device is not None
        zt = torch.randn(b, self.nz).to(device)
        for i in reversed(range(0, self.n_interval)):
            i_tensor = torch.ones(b, dtype=torch.float).to(device) * float(i)
            logsnr_t = logsnr_schedule_fn(i_tensor / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            logsnr_s = logsnr_schedule_fn(torch.clamp(i_tensor - 1.0, min=0.0) / (self.n_interval - 1.), logsnr_min=self.logsnr_min, logsnr_max=self.logsnr_max)
            eps_pred = self.p(z=zt, logsnr=logsnr_t, xemb=None)
            logsnr_t = logsnr_t.reshape((b, 1))
            logsnr_s = logsnr_s.reshape((b, 1))
            pred_z = pred_x_from_eps(z=zt, eps=eps_pred, logsnr=logsnr_t)

            if i == 0:
                zt = pred_z
            else:
                z_s_dist = diffusion_reverse(x=pred_z, z_t=zt, logsnr_s=logsnr_s, logsnr_t=logsnr_t, pred_var_type=self.var_type)
                eps = torch.randn_like(zt)
                zt = z_s_dist['mean'] + z_s_dist['std'] * eps

        return zt   
        
    def calculate_loss(self, x=None, z=None, mask=None):
        # given inferred z train unconditional diffusion model
        assert x is None and mask is None and z is not None
        u = torch.rand(len(z)).to(z.device)
        logsnr = logsnr_schedule_fn(u, logsnr_max=self.logsnr_max, logsnr_min=self.logsnr_min)
        zt_dist = diffusion_forward(z, logsnr=logsnr.reshape(len(z), 1))
        eps = torch.randn_like(z)
        zt = zt_dist['mean'] + zt_dist['std'] * eps
        eps_pred = self.p(z=zt, logsnr=logsnr, xemb=None)
        assert eps.shape == eps_pred.shape == (len(z), self.nz)
        loss = 0.5 * torch.sum((eps - eps_pred) ** 2, dim=1)
        return loss




