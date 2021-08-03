from typing import Tuple, Optional

import signatory
import torch
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import math

from lib.augmentations import apply_augmentations, parse_augmentations, Basepoint
from lib.trainers.base import BaseTrainer
from torch import optim


class SigWGANTrainer(BaseTrainer):
    def __init__(self, G, lr, depth, x_real_rolled, augmentations, normalise_sig: bool = True, mask_rate=0.01, **kwargs):
        super(SigWGANTrainer, self).__init__(G=G, G_optimizer=optim.Adam(G.parameters(), lr=lr), **kwargs)
        self.sig_w1_metric = SigW1Metric(depth=depth, x_real=x_real_rolled, augmentations=augmentations, mask_rate=mask_rate, normalise=normalise_sig)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)

    def fit(self, device):
        self.G.to(device)
        best_loss = None
        for j in tqdm(range(self.n_gradient_steps)):
            self.G_optimizer.zero_grad()
            x_fake = self.G(
                batch_size=self.batch_size, n_lags=self.sig_w1_metric.n_lags, device=device
            )
            loss = self.sig_w1_metric(x_fake)
            loss.backward()
            best_loss = loss.item() if j==0 else best_loss
            if (j+1)%100 == 0: print("sig-w1 loss: {:1.2e}".format(loss.item()))
            self.G_optimizer.step()
            self.scheduler.step()
            self.losses_history['sig_w1_loss'].append(loss.item())
            self.evaluate(x_fake)
            #if loss < best_loss:
            #    best_G = deepcopy(self.G.state_dict())
            #    best_loss = loss
        
        self.G.load_state_dict(self.best_G) # we retrieve the best generator




class SigWGANTrainerDyadicWindows(BaseTrainer):
    def __init__(self, G, lr, depth, x_real_rolled, augmentations, mask_rate=0.01, q=3, **kwargs):
        super(SigWGANTrainerDyadicWindows, self).__init__(G=G, G_optimizer=optim.Adam(G.parameters(), lr=lr), **kwargs)
        self.n_lags = x_real_rolled.shape[1] 
        
        # we create sig-w1-metric for all the dyadic windows
        self.sig_w1_metric = defaultdict(list) # Dictionary that will have key-value pairs (j, sig_w1_metric on the 2^j windows of x_real_rolled)
        # we will only inlcude basepoint when the dyadic window includes the first step of the path
        aug_ = augmentations.copy()
        try:
            aug_.remove(Basepoint())
        except:
            pass

        for j in range(q+1):
            n_intervals = 2**j
            len_interval = x_real_rolled.shape[1] // n_intervals
            for i in range(n_intervals):
                aug = augmentations if i==0 else aug_
                ind_min = max(0, i*len_interval-1)
                if i < (n_intervals-1):
                    self.sig_w1_metric[j].append(SigW1Metric(depth=depth, x_real=x_real_rolled[:,ind_min: (i+1)*len_interval,:], augmentations=aug, mask_rate=mask_rate, normalise=True))
                else:
                    self.sig_w1_metric[j].append(SigW1Metric(depth=depth, x_real=x_real_rolled[:,ind_min:,:], augmentations=aug, mask_rate=mask_rate, normalise=True))
        
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)

    def fit(self, device):
        self.G.to(device)
        
        best_loss = 10
        
        for it in tqdm(range(self.n_gradient_steps)):
            self.G_optimizer.zero_grad()
            x_fake = self.G(
                batch_size=self.batch_size, n_lags=self.n_lags, device=device
            )
            loss = 0
            
            for j in self.sig_w1_metric.keys():
                # we calculate the sig-w1-metric on each of the 2^j intervals given by the dyadic windows
                len_interval = self.n_lags // 2**j
                for i, sig_w1_metric_ in enumerate(self.sig_w1_metric[j]):
                    ind_min = max(0, i*len_interval-1)
                    if i < len(self.sig_w1_metric[j]) - 1:
                        loss += sig_w1_metric_(x_fake[:,ind_min:(i+1)*len_interval,:])
                    else:
                        loss += sig_w1_metric_(x_fake[:,ind_min:self.n_lags,:])

            best_loss = loss.item() if it == 0 else best_loss
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5)
            if (it+1)%100 == 0: print("sig-w1 loss: {:1.2e}".format(loss.item()))
            self.G_optimizer.step()
            self.scheduler.step()
            self.losses_history['sig_w1_loss'].append(loss.item())
            self.evaluate(x_fake)
            if loss < best_loss:
                best_G = deepcopy(self.G.state_dict())
                best_loss = loss

        self.G.load_state_dict(best_G) # we retrieve the best generator



def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
    x_path_augmented = apply_augmentations(x_path, augmentations)
    expected_signature = signatory.signature(x_path_augmented, depth=depth).mean(0)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            expected_signature[count:count + dim**(i+1)] = expected_signature[count:count + dim**(i+1)] * math.factorial(i+1)
            count = count + dim**(i+1)
    return expected_signature


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()

def masked_rmse(x, y, mask_rate, device):
    mask = torch.FloatTensor(x.shape[0]).to(device).uniform_() > mask_rate
    mask = mask.int()
    return ((x - y).pow(2) * mask).mean().sqrt()


class SigW1Metric:
    def __init__(self, depth: int, x_real: torch.Tensor, mask_rate:float, augmentations: Optional[Tuple] = (), normalise: bool = True):
        assert len(x_real.shape) == 3, \
            'Path needs to be 3-dimensional. Received %s dimension(s).' % (len(x_real.shape),)

        self.augmentations = augmentations
        self.depth = depth
        self.n_lags = x_real.shape[1]
        self.mask_rate = mask_rate

        self.normalise = normalise
        self.expected_signature_mu = compute_expected_signature(x_real, depth, augmentations, normalise)
        

    def __call__(self, x_path_nu: torch.Tensor):
        """ Computes the SigW1 metric."""
        device = x_path_nu.device
        batch_size = x_path_nu.shape[0]
        #expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
        #expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
        #y = self.expected_signature_mu.to(device)
        #loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
        #loss = loss.sum()
        expected_signature_nu = compute_expected_signature(x_path_nu, self.depth, self.augmentations, self.normalise)
        loss = rmse(self.expected_signature_mu.to(device), expected_signature_nu)
        #loss = masked_rmse(self.expected_signature_mu.to(
        #    device), expected_signature_nu, self.mask_rate, device)
        return loss
