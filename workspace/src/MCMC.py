# ############################################################################
# Include addtional functions for MCMC inference
# ############################################################################
import torch
import torchvision
import pytorch_fid_wrapper as pfw

def sample_langevin_prior_z(z, netE, e_l_steps, e_l_step_size, e_l_with_noise, verbose=False):
    mystr = "Step/en/z_norm: "
    for i in range(e_l_steps):
        en = netE(z).sum()
        z_norm = 1.0 / 2.0 * torch.sum(z**2)
        z_grad = torch.autograd.grad(-en + z_norm, z)[0]

        z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * z_grad 
        if e_l_with_noise:
            z.data += e_l_step_size * torch.randn_like(z)

        if (i % 5 == 0 or i == e_l_steps - 1):
            mystr += "{}/{:.3f}/{:.3f}  ".format(i, en.item(), z_norm.item())
    if verbose:
        print("Log prior sampling.")
        print(mystr)
    z.requires_grad = False
    return z.detach()

def sample_langevin_post_z(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/en/z_norm/recons_loss: "
    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        en = netE(z).sum()
        z_norm = 1.0 / 2.0 * torch.sum(z**2)
        total_en = g_log_lkhd - en + z_norm
        z_grad = torch.autograd.grad(total_en, z)[0]

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
        if g_l_with_noise:
            z.data += g_l_step_size * torch.randn_like(z)
        mystr += "{}/{:.3f}/{:.3f}/{:.3f}  ".format(i, en.item(), z_norm.item(), g_log_lkhd.item())
    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    z.requires_grad = False
    return z.detach()

def sample_langevin_post_z_with_diffusion(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "
    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        en = netE.calculate_loss(z=z).sum()
        total_en = g_log_lkhd + en
        z_grad = torch.autograd.grad(total_en, z)[0]

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
        if g_l_with_noise:
            z.data += g_l_step_size * torch.randn_like(z)
        mystr += "{}/{:.3f}/{:.3f}  ".format(i, en.item(), g_log_lkhd.item())
    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    z.requires_grad = False
    return z.detach()

def sample_langevin_post_z_with_gaussian(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "
    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        en = 1.0 / 2.0 * torch.sum(z**2)
        total_en = g_log_lkhd + en
        z_grad = torch.autograd.grad(total_en, z)[0]

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
        if g_l_with_noise:
            z.data += g_l_step_size * torch.randn_like(z)
        mystr += "{}/{:.3f}/{:.3f}  ".format(i, en.item(), g_log_lkhd.item())
    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    z.requires_grad = False
    return z.detach()

def gen_samples(bs, nz, netE, netG, e_l_steps, e_l_step_size, e_l_with_noise):
    zk_prior = torch.randn(bs, nz).cuda()
    zk_prior.requires_grad = True
    zk_prior = sample_langevin_prior_z(z=zk_prior, netE=netE, e_l_steps=e_l_steps, e_l_step_size=e_l_step_size, e_l_with_noise=e_l_with_noise, verbose=False)
    with torch.no_grad():
        x = netG(zk_prior)
    return x

def calculate_fid(n_samples, nz, netE, netG, e_l_steps, e_l_step_size, e_l_with_noise, real_m, real_s, save_name):
    bs = 500
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

def gen_samples_with_diffusion_prior(b, device, netE, netG):
    with torch.no_grad():
        zk_prior = netE(x=None, b=b, device=device)
        x = netG(zk_prior)
    return x, zk_prior

def calculate_fid_with_diffusion_prior(n_samples, device, netE, netG, real_m, real_s, save_name):
    bs = 500
    fid_samples = []
        
    for i in range(n_samples // bs):
        cur_samples, _ = gen_samples_with_diffusion_prior(bs, device, netE, netG)
        fid_samples.append(cur_samples.detach().clone())
        
    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device="cuda:0")
    if save_name is not None:
        save_images = fid_samples[:64].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=8)
        
    return fid