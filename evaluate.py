"""
Evaluation of a trained generator.
"""
import os
import os.path as pt
from lib.utils import load_obj
import matplotlib.pyplot as plt
import numpy as np
from lib.networks import get_generator
from lib.plot import *
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from collections import defaultdict


def gather_experiment_objs(root, obj_name):
    container = dict()
    for dir in sorted(os.listdir(root)):
        if pt.isdir(pt.join(root, dir)) and not 'LogSig' in dir:
           filepath = pt.join(root, dir, obj_name)
           if pt.exists(filepath):
               container[dir] = load_obj(filepath)
    return container


def get_seed(experiment_dir):
    seed = int(experiment_dir.split('_')[-1])
    return seed


def compare_loss_development(experiment_dir, loss_type='sig_w1_loss'):
    container = gather_experiment_objs(experiment_dir, 'losses_history.pkl')
    for k, v in container.items():
        if len(v['sig_w1_loss']) >= 10:
            plt.plot(np.array(v['sig_w1_loss'])[..., None].mean(1), label=k)
    plt.legend()
    plt.show()


def logrtn(x):
    y = x.log()
    return y[:, 1:] - y[:, :-1]


def evaluate_generator(experiment_dir, batch_size=1000, device='cpu', foo = lambda x: x):
    generator_config = load_obj(pt.join(experiment_dir, 'generator_config.pkl'))
    generator_state_dict = load_obj(pt.join(experiment_dir, 'generator_state_dict.pt'))
    generator = get_generator(**generator_config)
    generator.load_state_dict(generator_state_dict)

    data_config = load_obj(pt.join(experiment_dir, 'data_config.pkl'))
    x_real = torch.from_numpy(load_obj(pt.join(experiment_dir, 'x_real_test.pkl'))).detach()

    n_lags = data_config['n_lags']

    with torch.no_grad():
        x_fake = generator(batch_size, n_lags, device)
        x_fake = foo(x_fake)

    plot_summary(x_real=x_real, x_fake=x_fake)
    plt.savefig(pt.join(experiment_dir, 'comparison.png'))
    plt.close()

    # compute_discriminative_score(generator, x_real)
    for i in range(x_real.shape[2]):
        fig = plot_hists_marginals(x_real=x_real[...,i:i+1], x_fake=x_fake[...,i:i+1])
        fig.savefig(pt.join(experiment_dir, 'hists_marginals_dim{}.pdf'.format(i)))
        plt.close()

def compute_discriminative_score(generator, x_real,
                                 n_generated_paths=1024,
                                 n_discriminator_steps=2048, hidden_size=64, num_layers=3):

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(Discriminator, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.linear(h)

    model = Discriminator(x_real.shape[-1], hidden_size, num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_lags = x_real.shape[1]
    loss_history = defaultdict(list)

    for _ in tqdm(n_discriminator_steps):
        opt.zero_grad()

        with torch.no_grad():
            x_fake = generator(n_lags, n_generated_paths)

        d_fake = model(x_fake)
        d_real = model(x_real)

        targets = d_fake.new_full(size=d_fake.size(), fill_value=0.)
        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, targets)
        targets = d_fake.new_full(size=d_real.size(), fill_value=1.)
        d_loss_real = F.binary_cross_entropy_with_logits(d_real, targets)
        d_loss = d_loss_fake + d_loss_real
        d_loss.backward()
        opt.step()

        loss_history['d_loss_fake'].append(d_loss_fake.item())
        loss_history['d_loss_real'].append(d_loss_real.item())

    return loss_history
