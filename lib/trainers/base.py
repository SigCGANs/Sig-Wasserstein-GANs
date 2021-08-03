import copy
from collections import defaultdict
import time

import torch

from lib.utils import to_numpy


class BaseTrainer:
    def __init__(self, batch_size, G, G_optimizer, test_metrics_train, test_metrics_test, n_gradient_steps, foo=lambda x: x):
        self.batch_size = batch_size

        self.G = G
        self.G_optimizer = G_optimizer
        self.n_gradient_steps = n_gradient_steps

        self.losses_history = defaultdict(list)
        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test
        self.foo = foo
        
        self.init_time = time.time()

        self.best_G = copy.deepcopy(G.state_dict())
        self.best_cov_err = None

    def evaluate(self, x_fake):
        self.losses_history['time'].append(time.time() - self.init_time)
        with torch.no_grad():
            for test_metric in self.test_metrics_train:
                test_metric(x_fake)
                loss = to_numpy(test_metric.loss_componentwise)
                if len(loss.shape) == 1:
                    loss = loss[..., None]
                self.losses_history[test_metric.name + '_train'].append(loss)
            for test_metric in self.test_metrics_test:
                test_metric(x_fake)
                loss = to_numpy(test_metric.loss_componentwise)
                if len(loss.shape) == 1:
                    loss = loss[..., None]
                self.losses_history[test_metric.name + '_test'].append(loss)
        self.best_cov_err = self.losses_history['covariance_test'][-1].item() if self.best_cov_err == None else self.best_cov_err
        if self.losses_history['covariance_test'][-1].item() < self.best_cov_err:
            self.best_G = copy.deepcopy(self.G.state_dict())
            self.best_cov_err = self.losses_history['covariance_test'][-1].item()
