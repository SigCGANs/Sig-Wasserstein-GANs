import torch
from torch import autograd
from tqdm import tqdm
from typing import Tuple

from lib.trainers.base import BaseTrainer
from lib.utils import sample_indices
from lib.augmentations import apply_augmentations


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class WGANTrainer(BaseTrainer):
    def __init__(self, D, G, discriminator_steps_per_generator_step,
            lr_discriminator, lr_generator, x_real: torch.Tensor, reg_param=10.,
                 **kwargs):
        if kwargs.get('augmentations') is not None:
            self.augmentations = kwargs['augmentations']
            del kwargs['augmentations']
        else:
            self.augmentations = None
        super(WGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(G.parameters(), lr=lr_generator, betas=(0, 0.9)),
            **kwargs
        )
        self.D_steps_per_G_step = discriminator_steps_per_generator_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(D.parameters(), lr=lr_discriminator, betas=(0, 0.9))  # Using TTUR

        self.reg_param = reg_param
        if self.augmentations is not None:
            self.x_real = apply_augmentations(x_real, self.augmentations)
        else:
            self.x_real = x_real

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)

        for _ in tqdm(range(self.n_gradient_steps)):
            self.step(device)

    def step(self, device):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_real_batch = self.x_real[indices].to(device)
            with torch.no_grad():
                x_fake = self.G(batch_size=self.batch_size, n_lags=self.x_real.shape[1], device=device)
                if self.augmentations is not None:
                    x_fake = apply_augmentations(x_fake, self.augmentations)

            D_loss_real, D_loss_fake, wgan_gp = self.D_trainstep(x_fake, x_real_batch)
            if i == 0:
                self.losses_history['D_loss_fake'].append(D_loss_fake)
                self.losses_history['D_loss_real'].append(D_loss_real)
                self.losses_history['D_loss'].append(D_loss_fake + D_loss_real)
                self.losses_history['WGAN_GP'].append(wgan_gp)
        G_loss = self.G_trainstep(device)
        self.losses_history['G_loss'].append(G_loss)

    def G_trainstep(self, device):
        x_fake = self.G(batch_size=self.batch_size, n_lags=self.x_real.shape[1], device=device)
        if self.augmentations is not None:
            x_fake = apply_augmentations(x_fake, self.augmentations)

        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        self.D.train()
        G_loss = self.compute_loss(d_fake, 1)
        G_loss.backward()
        self.G_optimizer.step()
        self.evaluate(x_fake)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Compute regularizer on fake / real
        dloss = dloss_fake + dloss_real
        with torch.backends.cudnn.flags(enabled=False):
            wgan_gp = self.reg_param * self.wgan_gp_reg(x_real, x_fake)
        total_loss = dloss + wgan_gp
        total_loss.backward()

        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return dloss_real.item(), dloss_fake.item(), wgan_gp.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        return (2. * target - 1.) * d_out.mean()

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg
