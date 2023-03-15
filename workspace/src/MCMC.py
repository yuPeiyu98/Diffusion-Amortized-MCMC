# ############################################################################
# Include addtional functions for MCMC inference
# ############################################################################
import torch
import torch.nn.functional as F
import torchvision
import pytorch_fid_wrapper as pfw

from .diffusion_helper_func import logsnr_schedule_fn

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

def sample_langevin_post_z_with_prior(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "

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
    return z.detach()

def sample_langevin_post_z_with_prior_test(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "

    loss = torch.ones(z.size(0), device=z.device) * 100

    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        total_en = g_log_lkhd
        z_grad = torch.autograd.grad(total_en, z)[0]

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
        z.data += g_l_step_size * torch.randn_like(z)

        with torch.no_grad():
            x_ = netG(z)
            g_loss = torch.mean((x_ - x) ** 2, dim=[1,2,3])
            loss = torch.min(loss, g_loss)

        mystr += "{}/{:.3f}/{:.3f}/{:.3f}/{:.8f}  ".format(
            i, en.item(), g_log_lkhd.item(), 
            z_n.item(), z_grad.mean().item())
    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    return z.detach(), loss

def sample_langevin_post_z_with_prior_mh(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "
    g_l_step_size_ = g_l_step_size

    ##### initial k0 state
    x_hat = netG(z)
    g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2, dim=[1, 2, 3])
    z_n = 1.0 / 2.0 * torch.sum(z**2, dim=1) 
    en = netE(z)
    total_en = g_log_lkhd + en + z_n
    z_grad = torch.autograd.grad(total_en.sum(), z)[0]

    z_0 = z.clone().detach()
    z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
    noise = g_l_step_size * torch.randn_like(z)
    z_t = z + noise

    z = z.detach()
    for i in range(g_l_steps):
        ##### mh adjustment
        log_q_k1_k = - 0.5 * (noise ** 2).sum(dim=1) / (g_l_step_size ** 2)

        # k1 state
        x_hat_ = netG(z_t)
        g_log_lkhd_ = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat_ - x) ** 2, dim=[1, 2, 3])
        z_n_ = 1.0 / 2.0 * torch.sum(z_t**2, dim=1) 
        en_ = netE(z_t)
        total_en_ = g_log_lkhd_ + en_ + z_n_
        z_grad_ = torch.autograd.grad(total_en_.sum(), z_t)[0]

        z_0_ = z_t.clone().detach()
        z_ = z_t - 0.5 * g_l_step_size * g_l_step_size * z_grad_
        log_q_k_k1 = - 0.5 * torch.sum((z_0 - z_) ** 2, dim=1) / (g_l_step_size ** 2)

        # ad-rj
        prop = - total_en_ + log_q_k_k1 + total_en - log_q_k1_k
        p_acc = torch.minimum(torch.ones_like(prop), torch.exp(prop))
        replace_idx = p_acc >= torch.rand_like(p_acc)
        acc_rate = torch.mean(replace_idx.float()).item()

        ##### update status
        if acc_rate < 0.574:
            g_l_step_size *= 2
        else:
            g_l_step_size *= 0.5

        if acc_rate < .1:
            z_0 = torch.randn_like(z)
            z = torch.randn_like(z)
            noise = g_l_step_size * torch.randn_like(z)
            z_t = torch.randn_like(z, requires_grad=True)

            g_l_step_size = g_l_step_size_
        else:
            z_0[replace_idx] = z_0_[replace_idx]
            z[replace_idx] = z_[replace_idx].detach()
            noise = g_l_step_size * torch.randn_like(z)
            z_t = (z + noise).requires_grad_(True)

        mystr += "{}/{:.3f}/{:.3f}/{:.3f}/{:.8f}/{:.3f}  ".format(
            i, en_.mean().item(), g_log_lkhd_.mean().item(), 
            z_n_.mean().item(), z_grad_.mean().item(), acc_rate)

    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    return z_0.detach()

def sample_langevin_post_z_with_gaussian(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "

    i_tensor = torch.ones(z.size(0), dtype=torch.float, device=z.device)
    # xemb = torch.zeros(size=(z.size(0), netE.nxemb), device=z.device)
    xemb = netE.xemb.expand(z.size(0), -1)
    logsnr_t = logsnr_schedule_fn(i_tensor / (netE.n_interval - 1.), logsnr_min=netE.logsnr_min, logsnr_max=netE.logsnr_max)
    
    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        en = 1.0 / 2.0 * torch.sum(z**2)
        total_en = g_log_lkhd + en
        z_grad = torch.autograd.grad(total_en, z)[0]

        # prior grad
        with torch.no_grad():
            eps_pred = netE.p(z=z, logsnr=logsnr_t, xemb=xemb)
        # zp_grad = eps_pred # * torch.rsqrt(1. + torch.exp(logsnr_t)).unsqueeze(1)
        zp_grad = eps_pred / torch.rsqrt(1. + torch.exp(logsnr_t)).unsqueeze(1)
        zp_grad_norm = torch.linalg.norm(zp_grad, dim=1, keepdim=True)
        mask = (zp_grad_norm > 100.0) * 1.0
        zp_grad = mask * zp_grad / zp_grad_norm + (1 - mask) * zp_grad

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * (z_grad + zp_grad)
        if g_l_with_noise:
            z.data += g_l_step_size * torch.randn_like(z)
        mystr += "{}/{:.3f}/{:.3f}/{:.8f}/{:.8f}/{:.8f}  ".format(
            i, en.item(), g_log_lkhd.item(), 
            z.mean().item(), (z_grad - z).mean().item(), zp_grad.mean().item())
    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    return z.detach()

def sample_hmc_post_z_with_gaussian(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "

    i_tensor = torch.ones(z.size(0), dtype=torch.float, device=z.device)
    # xemb = torch.zeros(size=(z.size(0), netE.nxemb), device=z.device)
    xemb = netE.xemb.expand(z.size(0), -1)
    logsnr_t = logsnr_schedule_fn(i_tensor / (netE.n_interval - 1.), logsnr_min=netE.logsnr_min, logsnr_max=netE.logsnr_max)
    
    th = 0.651
    L = 3
    step_sz = g_l_step_size
    step_mul = 1.02

    z.requires_grad = False

    def en(z_):
        x_hat = netG(z_)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        en = 1.0 / 2.0 * torch.sum(z_**2)
        total_en = g_log_lkhd + en

        return total_en

    def grad(z_):
        total_en = en(z_)
        z_grad = torch.autograd.grad(total_en, z_)[0]

        # prior grad
        with torch.no_grad():
            eps_pred = netE.p(z=z_, logsnr=logsnr_t, xemb=xemb)
        # zp_grad = eps_pred # * torch.rsqrt(1. + torch.exp(logsnr_t)).unsqueeze(1)
        zp_grad = eps_pred / torch.rsqrt(1. + torch.exp(logsnr_t)).unsqueeze(1)
        zp_grad_norm = torch.linalg.norm(zp_grad, dim=1, keepdim=True)
        mask = (zp_grad_norm > 100.0) * 1.0
        zp_grad = mask * zp_grad / zp_grad_norm + (1 - mask) * zp_grad

        return z_grad + zp_grad

    for k in range(g_l_steps):
        # initialize the dynamics and the momentum
        p0, z_ = torch.randn_like(z), z.clone().detach().requires_grad_(True)
        # first half-step update for the momentum and 
        # the full step update for the data
        p = p0 - 0.5 * step_sz * grad(z_)
        z_ = z_ + step_sz * p
        for __ in range(L):
            p = p + step_sz * grad(z_)
            z_ = z_ + step_sz * p
        # the last half-step update for the momentum    
        p = p + step_sz * grad(z_)
        
        # Metropolis-Hastings Correction
        en_0, en_e = en(z), en(z_)
        H0 = en_0 + 0.5 * torch.sum(p0.square().view(p0.size(0), -1), 1)
        H1 = en_e + 0.5 * torch.sum(p.square().view(p.size(0), -1), 1)    
        p_acc = torch.minimum(torch.ones_like(H0), torch.exp(H0 - H1))
        replace_idx = p_acc > torch.rand_like(p_acc)
        z[replace_idx] = z_[replace_idx].detach().clone()

        acc_rate = torch.mean(replace_idx.float()).item()

        mystr += "{}/{:.3f}/{:.3f}/{:.8f}/{:.8f}/{:.8f}  ".format(
                k, en_e.item(), en_0.item(), 
                z.mean().item(), step_sz, acc_rate)

        if acc_rate > th:
            step_sz *= step_mul
        else:
            step_sz /= step_mul
    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    return z.detach()

def sample_consensus_post_z_with_gaussian(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "

    (B, c), N = z.size(), 2
    z  = z.reshape(B, 1, c)
    z_ = torch.randn(size=(B, N, c), device=z.device, requires_grad=True)
    z  = torch.cat([z, z_], dim=1)
    z  = z.reshape(B*(N + 1), c)

    # beta = 40

    __, d, h, w = x.size()
    x = x.unsqueeze(1).expand(-1, N + 1, -1, -1, -1)
    x = x.reshape(B * (N + 1), d, h, w)

    for i in range(g_l_steps):
        # z  = z.reshape(B*(N + 1), c)

        x_hat = netG(z)

        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2, dim=[1, 2, 3])
        en = 1.0 / 2.0 * torch.sum(z**2, dim=1)
        total_en = g_log_lkhd + en

        # w = (-beta * total_en).softmax(dim=1)
        # z_star = (w * z).sum(dim=1, keepdim=True)
        # z_diff = z - z_star
        # n = torch.randn_like(z_diff)
        # z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_diff + 0.9 * g_l_step_size * z_diff * n
        # mystr += "{}/{:.3f}/{:.3f}/{:.8f}  ".format(i, en.mean().item(), g_log_lkhd.mean().item(), z.mean().item())

        z_grad = torch.autograd.grad(total_en.sum(), z)[0]

        z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * z_grad
            
        mystr += "{}/{:.3f}/{:.3f}/{:.8f}/{:.8f}  ".format(
            i, en.sum().item(), g_log_lkhd.sum().item(), z.abs().max().item(), (z_grad - z).abs().max().item())

    if verbose:
        print("Log posterior sampling.")
        print(mystr)
    with torch.no_grad():
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2, dim=[1, 2, 3])
        en = 1.0 / 2.0 * torch.sum(z**2, dim=1)
        total_en = g_log_lkhd + en
    idx = total_en.reshape(B, N + 1).argmin(dim=1)
    idx = idx.reshape(B, 1, 1).expand(-1, -1, c)
    z  = z.reshape(B, N + 1, c).gather(1, idx)
    return z.detach().squeeze(1)

def sample_langevin_post_z_with_diffgrad(z, x, netG, netE, g_l_steps, g_llhd_sigma, g_l_with_noise, g_l_step_size, verbose = False):
    mystr = "Step/cross_entropy/recons_loss: "
    b = len(x)
    device = x.device

    # if x is not None:
    #     xemb = netE.encoder(x)
    # else:
    #     xemb = torch.zeros(b, netE.nxemb).to(device)

    xemb = torch.zeros(b, netE.nxemb).to(device)

    for i in range(g_l_steps):
        x_hat = netG(z)
        g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * torch.sum((x_hat - x) ** 2)
        total_en = g_log_lkhd
        z_grad = torch.autograd.grad(total_en, z)[0] 

        # prior grad
        i_tensor = torch.zeros(b, dtype=torch.float).to(device)
        logsnr_t = logsnr_schedule_fn(i_tensor / (netE.n_interval - 1.), logsnr_min=netE.logsnr_min, logsnr_max=netE.logsnr_max)
        with torch.no_grad():
            eps_pred = netE.p(z=z, logsnr=logsnr_t, xemb=xemb)
        zp_grad = - eps_pred / torch.rsqrt(1. + torch.exp(logsnr_t))

        z.data = z.data + 0.5 * g_l_step_size * g_l_step_size * (-z_grad + zp_grad)
        if g_l_with_noise:
            z.data += g_l_step_size * torch.randn_like(z)
        mystr += "{}/{:.3f}/{:.3f}  ".format(i, zp_grad.mean().item(), g_log_lkhd.item())
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

def gen_samples_with_diffusion_prior(b, device, netQ, netG):
    with torch.no_grad():
        zk_prior = netQ(x=None, b=b, device=device)
        x = netG(zk_prior)
    return x, zk_prior

def gen_samples_with_diffusion_prior_E(b, device, netQ, netG, netE):
    K = 10
    with torch.no_grad():
        zk_prior = netQ(x=None, b=b * K, device=device).reshape(b, K, -1)
        logit = -netE(zk_prior) # (b, K)
        # i = F.gumbel_softmax(logit, dim=1, hard=True).argmax(dim=1).unsqueeze(1)
        i = F.softmax(logit, dim=1).argmax(dim=1).unsqueeze(1)
        i = i.unsqueeze(1).expand(-1, -1, zk_prior.size(-1))
        zk_prior = torch.gather(zk_prior, dim=1, index=i).squeeze(1)
        x = netG(zk_prior)
    return x, zk_prior

def calculate_fid_with_diffusion_prior(n_samples, device, netQ, netG, netE, real_m, real_s, save_name):
    bs = 500
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

def calculate_fid_with_diffusion_prior_E(n_samples, device, netQ, netG, netE, real_m, real_s, save_name):
    bs = 500
    fid_samples = []
        
    for i in range(n_samples // bs):
        cur_samples, _ = gen_samples_with_diffusion_prior_E(bs, device, netQ, netG, netE)
        fid_samples.append(cur_samples.detach().clone())
        
    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device="cuda:0")
    if save_name is not None:
        save_images = fid_samples[:64].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=8)
        
    return fid
