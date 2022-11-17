# ############################################################################
# Include addtional functions needed by diffusion model
# ############################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class log1mexp(torch.autograd.Function):
    # From James Townsend's PixelCNN++ code
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.where(input > np.log(2.), torch.log1p(-torch.exp(-input)), torch.log(-torch.expm1(-input)))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output / torch.expm1(input)

mylog1mexp = log1mexp.apply

def pred_x_from_eps(z, eps, logsnr):
    #print('coeff1', torch.sqrt(1. + torch.exp(-logsnr)).max(), torch.sqrt(1. + torch.exp(-logsnr)).min())
    #print('coeff2', torch.rsqrt(1. + torch.exp(logsnr)).max(), torch.rsqrt(1. + torch.exp(logsnr)).min())
    return torch.sqrt(1. + torch.exp(-logsnr)) * (z - eps * torch.rsqrt(1. + torch.exp(logsnr)))

def logsnr_schedule_fn(t, logsnr_min=-20, logsnr_max=20):
    # -2log(tan(b)) == logsnr_max => b == arctan(exp(-0.5*logsnr_max))
    # -2log(tan(pi/2*a + b)) == logsnr_min
    #     => a == (arctan(exp(-0.5*logsnr_min))-b)*2/pi
    logsnr_min_tensor = logsnr_min * torch.ones_like(t)
    logsnr_max_tensor = logsnr_max * torch.ones_like(t)
    b = torch.arctan(torch.exp(-0.5 * logsnr_max_tensor))
    a = torch.arctan(torch.exp(-0.5 * logsnr_min_tensor)) - b
    #print(a[0], b[0], torch.exp(-0.5 * logsnr_max_tensor[0]), torch.exp(-0.5 * logsnr_min_tensor[0]))
    return -2. * torch.log(torch.tan(a * t + b))

def diffusion_reverse(x, z_t, logsnr_s, logsnr_t, pred_var_type='small'):
    alpha_st = torch.sqrt((1. + torch.exp(-logsnr_t)) / (1. + torch.exp(-logsnr_s)))
    alpha_s = torch.sqrt(F.sigmoid(logsnr_s))
    r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
    one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
    log_one_minus_r = mylog1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))
    mean = r * alpha_st * z_t + one_minus_r * alpha_s * x
    if pred_var_type == 'large':
        var = one_minus_r * F.sigmoid(-logsnr_t)
        logvar = log_one_minus_r + torch.log(F.sigmoid(-logsnr_t))
    elif pred_var_type == 'small':
        a_t = F.sigmoid(logsnr_t)
        a_tminus1 = F.sigmoid(logsnr_s)
        beta_t = (1 - a_t / a_tminus1)
        var = (1.0 - a_tminus1) / (1.0 - a_t) * beta_t
        logvar = torch.log(var)
    else:
        raise NotImplemented
    return {'mean': mean, 'std': torch.sqrt(var), 'var': var, 'logvar': logvar}

def diffusion_forward(x, logsnr):
    return {
        'mean': x * torch.sqrt(F.sigmoid(logsnr)),
        'std': torch.sqrt(F.sigmoid(-logsnr)),
        'var': F.sigmoid(-logsnr),
        'logvar': torch.log(F.sigmoid(-logsnr))
    }

def denoise_true(z, x0, logsnr_t, logsnr_tminus1):
    z_tminus1_dist = diffusion_reverse(x=x0, z_t=z, logsnr_s=logsnr_tminus1.reshape(len(z), 1), logsnr_t=logsnr_t.reshape(len(z), 1))
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    beta_t = (1 - a_t / a_tminus1)
    std = torch.sqrt((1.0 - a_tminus1) / (1.0 - a_t) * beta_t).reshape((len(z), 1))
    sample_x = z_tminus1_dist['mean'] + std * torch.randn_like(z)
    return sample_x
