import torch
import torch.nn as nn
import numpy as np
import signatory
from abc import abstractmethod

from lib.networks.resfnn import ResFNN
from lib.augmentations import *
from lib.utils import sample_indices
        

class W1_dist():

    def __init__(self, dataset1, dataset2, critic_hidden_sizes=[20,20], lambda_reg=100, lr=0.001):

        self.d = dataset1.shape[2] # path dimension
        self.p = dataset1.shape[1] # path length
        # Generators
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # critic
        self.C = ResFNN(input_dim=self._input_dim(), output_dim=1, hidden_dims=critic_hidden_sizes)
        # optimizers and schedulers
        self.C_optimizer = torch.optim.RMSprop(self.C.parameters(), lr=lr)
        self.C_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.C_optimizer, milestones=[1000], gamma=0.5)
        
        # regularization coefficient
        self.lambda_reg = lambda_reg
    
    @abstractmethod
    def _input_dim(self):
        ...

    @abstractmethod
    def _get_input(self,x):
        """
        Input given to critic. It will depend on the space where we are calculating the W1-dist (e.g. Path space, or SigSpace)
        Parameters
        ----------
        x: torch.Tensor
            tensor of shape (batch_size, L, d)
        Returns
        -------
        x_out: torch.Tensor
            Tensor of shape (batch_size, d')
        """
        ...
    
    def to_device(self, device):
        self.C.to(device)
    
    
    def update_critic(self, batch_size = 100):
        """
        Update critic using Wasserstein distance + regularizer term to enforce 1-Lipschitz of critic
        
        Parameters
        ----------
        x0 : torch.Tensor
            Starting point. Tensor of shape (N, self.d) where N is batch size

        """
        device = self.dataset1.device
        self.C_optimizer.zero_grad()
        # generate paths
        indices = sample_indices(self.dataset1.shape[0], batch_size).to(device)
        x_fake = self.dataset1[indices] # (batch_size, self.p, self.d)
        x_real = self.dataset2[indices] # (batch_size, self.p, self.d)
        # W1-dist
        x_real = self._get_input(x_real) 
        critic_x_real = self.C(x_real) # (batch_size,1)
        x_fake = self._get_input(x_fake)
        critic_x_fake = self.C(x_fake) # (batch_size, 1)
        neg_w1_dist = critic_x_fake.mean() - critic_x_real.mean() # we minimise over 1-Lip functions, so it will be the negative of the w1 dist

        # reg term to enforce self.C to be 1-Lipschitz
        eps = torch.rand(batch_size, 1, device=x_real.device)
        x_interp = (1-eps) * x_real + eps * x_fake
        x_interp = x_interp.requires_grad_(True)
        critic_x_interp = self.C(x_interp) # (batch_size, 1)
        path_derivative = torch.autograd.grad(critic_x_interp.sum(), x_interp,
                retain_graph=True,
                create_graph=True,
                only_inputs=True)[0]
        reg_term = torch.mean((torch.norm(path_derivative, p=2, dim=1)-1)**2)

        # loss function
        loss = neg_w1_dist + self.lambda_reg*reg_term
        loss.backward()
        self.C_optimizer.step()
        self.C_scheduler.step()
        return neg_w1_dist.mean().item(), reg_term.mean().item()

    def get_dist(self, batch_size):
        neg_w1_dist, _ = self.update_critic(batch_size)
        return -neg_w1_dist


class W1_dist_PathSpace(W1_dist):
    """
    Class that finds the Wasserstein1 distance between two paths
    """

    def _input_dim(self):
        return self.d * self.p
    
    def _get_input(self,x):
        """
        Parameters
        ----------
        x: torch.Tensor
            tensor of shape (batch_size, L, d)
        Returns
        -------
        x_out: torch.Tensor
            Tensor of shape (batch_size, d')
        """
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class W1_dist_SigSpace(W1_dist):

    def __init__(self, dataset1, dataset2, critic_hidden_sizes, lambda_reg=100, lr=0.001, depth=3, augmentations=(Scale(0.3), LeadLag(with_time=True),)):
        
        self.augmentations = augmentations
        self.depth = depth
        super().__init__(dataset1, dataset2, critic_hidden_sizes,lambda_reg,lr)
    
    def _input_dim(self):
        channels = self.d
        for aug in self.augmentations:
            if isinstance(aug,LeadLag):
                channels = channels*2
            if hasattr(aug, 'with_time'):
                channels = channels+1 if aug.with_time else channels
        return signatory.signature_channels(channels=channels, depth=self.depth)
    
    def _get_input(self, x):
        x_augmented = apply_augmentations(x, self.augmentations)
        return signatory.signature(x_augmented, depth=self.depth, basepoint=True)


