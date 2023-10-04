##############################################################################
# Include addtional functions for MCMC inference
##############################################################################

import torch
import torch.nn.functional as F
import torchvision
import pytorch_fid_wrapper as pfw

from .diffusion_helper_func import logsnr_schedule_fn

def set_requires_grad(nets, requires_grad=False):
    """ Set requies_grad=False for all the networks to 
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

def sample_langevin_prior_z(z, netE, e_l_steps, e_l_step_size, e_l_with_noise, verbose=False):
    mystr = "Step/en/z_norm: "

    set_requires_grad(netE, requires_grad=False)
    for i in range(e_l_steps):
        en = netE(z).sum()
        z_norm = 1.0 / 2.0 * torch.sum(z**2)
        z_grad = torch.autograd.grad(en + z_norm, z)[0]

        z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * z_grad 
        if e_l_with_noise:
            z.data += e_l_step_size * torch.randn_like(z)

        if (i % 5 == 0 or i == e_l_steps - 1):
            mystr += "{}/{:.3f}/{:.3f}  ".format(i, en.item(), z_norm.item())
    if verbose:
        print("Log prior sampling.")
        print(mystr)
    set_requires_grad(netE, requires_grad=True)
    return z.detach()

def sample_langevin_post_z_with_prior(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "

    set_requires_grad(netG, requires_grad=False)
    set_requires_grad(netE, requires_grad=False)

    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        z_n = 1.0 / 2.0 * torch.sum(z**2) 
        en = netE(z).sum()
        total_en = g_log_lkhd + en + z_n
        z_grad = torch.autograd.grad(total_en, z)[0]

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
        if g_l_with_noise:
            z.data += g_l_step_size * torch.randn_like(z)
        mystr += "{}/{:.3f}/{:.3f}/{:.3f}/{:.8f}  ".format(
            i, en.item(), g_log_lkhd.item(), 
            z_n.item(), z_grad.mean().item())
    if verbose:
        print("Log posterior sampling.")
        print(mystr)

    set_requires_grad(netG, requires_grad=True)
    set_requires_grad(netE, requires_grad=True)
    return z.detach()

def sample_invert_z(z, x, netG, netF, netE, g_l_steps, g_l_step_size, verbose = False):
    mystr = "Step/recon_loss/feat_loss: "

    set_requires_grad(netG, requires_grad=False)
    set_requires_grad(netF, requires_grad=False)
    set_requires_grad(netE, requires_grad=False)

    with torch.no_grad():
        x_hat = netG(z)
        g_log_lkhd = torch.mean((x_hat - x) ** 2, dim=[1,2,3])
        m = torch.isnan(g_log_lkhd).unsqueeze(1).expand(z.size(0), z.size(1))

        t = torch.randn(x.size(0), 512).cuda()
        w = netG.net.mapping(t, l=None)
        w = netG.net.truncation(w)
        w = w.reshape(x.size(0), -1)

    z_ = torch.where(m, w, z)
    z = z_.detach().clone()
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=g_l_step_size)

    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = torch.mean((x_hat - x) ** 2, dim=[1,2,3]).sum()
        f_l = torch.mean((netF(x) - netF(x_hat)) ** 2, dim=[1,2,3]).sum() 
        total_en = g_log_lkhd * 1.5 + f_l * 5e-5 

        optimizer.zero_grad()
        total_en.backward()
        optimizer.step()

        mystr += "{}/{:.3f}/{:.3f}/{:.3f} ".format(
            i, g_log_lkhd.item(), f_l.item(), 0)
    
    if verbose:
        print("Log posterior sampling.")
        print(mystr)

    set_requires_grad(netE, requires_grad=True)
    return z.detach()

def gen_samples(bs, nz, netE, netG, e_l_steps, e_l_step_size, e_l_with_noise):
    zk_prior = torch.randn(bs, nz).cuda()
    zk_prior.requires_grad = True
    
    zk_prior = _hmc_prior_sampler(
        z=zk_prior, netE=netE, e_l_steps=e_l_steps, e_l_step_size=e_l_step_size, e_l_with_noise=e_l_with_noise)
    with torch.no_grad():
        x = netG(zk_prior)
    return x

def calculate_fid(n_samples, nz, netE, netG, e_l_steps, e_l_step_size, e_l_with_noise, real_m, real_s, save_name, bs=500):
    fid_samples = []
        
    for i in range(n_samples // bs):
        cur_samples = gen_samples(bs, nz, netE, netG, e_l_steps, e_l_step_size, e_l_with_noise)
        fid_samples.append(cur_samples.detach().clone())
        
    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device="cuda:0")
    if save_name is not None:
        save_images = fid_samples[:64].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=8)
        
    return fid

def gen_samples_with_diffusion_prior(b, device, netQ, netG):
    with torch.no_grad():
        zk_prior = netQ(x=None, b=b, device=device)
        x = netG(zk_prior)
    return x, zk_prior

def calculate_fid_with_diffusion_prior(n_samples, device, netQ, netG, netE, real_m, real_s, save_name, bs=500):
    fid_samples = []
        
    for i in range(n_samples // bs):
        cur_samples, _ = gen_samples_with_diffusion_prior(bs, device, netQ, netG)
        fid_samples.append(cur_samples.detach().clone())
        
    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device="cuda:0")
    if save_name is not None:
        save_images = fid_samples[:64].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=8)
        
    return fid

def calculate_fid_with_samples(fid_samples, real_m, real_s, save_name):
    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device="cuda:0")
    if save_name is not None:
        save_images = fid_samples[:64].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=8)
        
    return fid
