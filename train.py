"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import os

from os import path as pt
from typing import Optional
import argparse

from lib.augmentations import parse_augmentations
from lib.networks import get_generator, get_discriminator
from lib.utils import to_numpy, set_seed, save_obj, load_obj
from lib.trainers.sig_wgan import SigWGANTrainer
from lib.trainers.wgan import WGANTrainer
from lib.test_metrics import get_standard_test_metrics
from lib.datasets import rolling_window, get_dataset, train_test_split
from lib.trainers.sig_wgan import compute_expected_signature
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate import evaluate_generator


def plot_signature(sig):
    plt.plot(to_numpy(sig).T, 'o')


def plot_test_metrics(test_metrics, losses_history, mode):
    fig, axes = plt.subplots(len(test_metrics), 1, figsize=(10, 8))
    for i, test_metric in enumerate(test_metrics):
        name = test_metric.name
        loss = losses_history[name + '_' + mode]
        try:
            loss = np.concatenate(loss, 1).T
        except:
            loss = np.array(loss)
        axes[i].plot(loss, label=name)
        axes[i].grid()
        axes[i].legend()
        axes[i].set_ylim(bottom=0.)
        if i == len(test_metrics):
            axes[i].set_xlabel('Number of generator weight updates')


def main(
        data_config: dict,
        dataset: str,
        experiment_dir: str,
        gan_algo: str,
        gan_config: dict,
        generator_config: dict,
        device: str = 'cpu',
        discriminator_config: Optional = None,
        seed: Optional[int] = 0
):
    """

    Full training procedure.
    Includes: initialising the dataset / generator / GAN and training the GAN.
    """

    n_lags = data_config.pop("n_lags")

    # Get / prepare dataset
    x_real = get_dataset(dataset, data_config, n_lags=n_lags)
    x_real = x_real.to(device)
    set_seed(seed)
    #x_real_rolled = rolling_window(x_real, n_lags, )
    x_real_rolled = x_real
    x_real_train, x_real_test = train_test_split(x_real_rolled, train_test_ratio=0.8)
    x_real_dim: int = x_real.shape[2]

    # Compute test metrics for train and test set
    test_metrics_train = get_standard_test_metrics(x_real_train)
    test_metrics_test = get_standard_test_metrics(x_real_test)

    # Get generator
    set_seed(seed)
    generator_config.update(output_dim=x_real_dim)
    G = get_generator(**generator_config).to(device)

    # Get GAN
    if gan_algo == 'SigWGAN':
        trainer = SigWGANTrainer(G=G,
                                 x_real_rolled=x_real_rolled,
                                 test_metrics_train=test_metrics_train,
                                 test_metrics_test=test_metrics_test,
                                 **gan_config
                                 )
    elif gan_algo == 'WGAN':
        set_seed(seed)
        discriminator_config.update(input_dim=x_real_dim * n_lags)
        D = get_discriminator(**discriminator_config)
        trainer = WGANTrainer(D, G,
                              x_real=x_real_rolled,
                              test_metrics_train=test_metrics_train,
                              test_metrics_test=test_metrics_test,
                              **gan_config
      )
    else:
        raise NotImplementedError()

    # Start training
    set_seed(seed)
    trainer.fit(device=device)

    # Store relevant training results
    save_obj(to_numpy(x_real), pt.join(experiment_dir, 'x_real.pkl'))
    save_obj(to_numpy(x_real_test), pt.join(experiment_dir, 'x_real_test.pkl'))
    save_obj(to_numpy(x_real_train), pt.join(experiment_dir, 'x_real_train.pkl'))
    save_obj(trainer.losses_history, pt.join(experiment_dir, 'losses_history.pkl'))  # dev of losses / metrics
    save_obj(trainer.G.state_dict(), pt.join(experiment_dir, 'generator_state_dict.pt'))
    save_obj(generator_config, pt.join(experiment_dir, 'generator_config.pkl'))

    if gan_algo == 'SigWGAN':
        plt.plot(trainer.losses_history['sig_w1_loss'], alpha=0.8)
        plt.grid()
        plt.yscale('log')
        plt.savefig(pt.join(experiment_dir, 'sig_loss.png'))
        plt.close()
    else:
        plt.plot(trainer.losses_history['D_loss_fake'])
        plt.plot(trainer.losses_history['D_loss_real'])
        plt.plot(np.array(trainer.losses_history['D_loss_real'])+np.array(trainer.losses_history['D_loss_fake']))
        plt.savefig(pt.join(experiment_dir, 'wgan_loss.png'))
        plt.close()

    plot_test_metrics(trainer.test_metrics_train, trainer.losses_history, 'train')
    plt.savefig(pt.join(experiment_dir, 'loss_development_train.png'))
    plt.close()

    plot_test_metrics(trainer.test_metrics_train, trainer.losses_history, 'test')
    plt.savefig(pt.join(experiment_dir, 'loss_development_test.png'))
    plt.close()

    with torch.no_grad():
        x_fake = G(1024, n_lags, device)

    for i in range(x_real_dim):
        plt.plot(to_numpy(x_fake[:250, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(pt.join(experiment_dir, 'x_fake.png'))
    plt.close()

    for i in range(x_real_dim):
        random_indices = torch.randint(0, x_real_rolled.shape[0], (250,))
        plt.plot(to_numpy(x_real_rolled[random_indices, :, i]).T, 'C%s' % i, alpha=0.1)
    plt.savefig(pt.join(experiment_dir, 'x_real.png'))
    plt.close()

    evaluate_generator(experiment_dir, batch_size=5000,)

    if gan_algo == 'WGAN':
        save_obj(trainer.D.state_dict(), pt.join(experiment_dir, 'discriminator_state_dict.pt'))
        save_obj(generator_config, pt.join(experiment_dir, 'discriminator_config.pkl'))
    else:
        plot_signature(trainer.sig_w1_metric.expected_signature_mu)
        plt.savefig(pt.join(experiment_dir, 'sig_real.png'))
        plt.close()

        plot_signature(trainer.sig_w1_metric.expected_signature_mu)
        plot_signature(compute_expected_signature(x_fake, trainer.sig_w1_metric.depth, trainer.sig_w1_metric.augmentations))
        plt.savefig(pt.join(experiment_dir, 'sig_real_fake.png'))
        plt.close()
        pass


def get_config_path(config, dataset):
    return './configs/{dataset}/{config}.json'.format(config=config, dataset=dataset)


def get_config_path_generator(config, dataset):
    return './configs/{dataset}/generator/{config}.json'.format(
        dataset=dataset, config=config
    )


def get_config_path_discriminator(config, dataset):
    return './configs/{dataset}/discriminator/{config}.json'.format(config=config, dataset=dataset)


def get_sigwgan_experiment_dir(dataset, generator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, seed=seed)


def get_wgan_experiment_dir(dataset, discriminator, generator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{discriminator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, discriminator=discriminator, seed=seed)


list_of_datasets = ('GBM', 'STOCKS', 'ECG')


def benchmark_wgan(
    datasets=list_of_datasets,
    discriminators=('ResFNN',),
    generators=('LSTM', 'NSDE',),
    n_seeds=10,
    device='cuda:0',
):
    """ Benchmark for WGAN. """
    seeds = list(range(n_seeds))

    grid = itertools.product(datasets, discriminators, generators, seeds)

    for dataset, discriminator, generator, seed in grid:
        data_config = load_obj(get_config_path(dataset, dataset))
        discriminator_config = load_obj(get_config_path_discriminator(discriminator, dataset))
        gan_config = load_obj(get_config_path('WGAN', dataset))
        generator_config = load_obj(get_config_path_generator(generator, dataset))

        if generator_config.get('augmentations') is not None:
            generator_config['augmentations'] = parse_augmentations(generator_config.get('augmentations'))

        if generator_config['generator_type'] == 'LogSigRNN':
            generator_config['n_lags'] = data_config['n_lags']

        experiment_dir = get_wgan_experiment_dir(dataset, discriminator, generator, 'WGAN', seed)

        if not pt.exists(experiment_dir):
            os.makedirs(experiment_dir)

        save_obj(data_config, pt.join(experiment_dir, 'data_config.pkl'))
        save_obj(discriminator_config, pt.join(experiment_dir, 'discriminator_config.pkl'))
        save_obj(gan_config, pt.join(experiment_dir, 'gan_config.pkl'))
        save_obj(generator_config, pt.join(experiment_dir, 'generator_config.pkl'))

        print('Training: %s' % experiment_dir.split('/')[-2:])

        main(
            dataset=dataset,
            data_config=data_config,
            device=device,
            experiment_dir=experiment_dir,
            gan_algo='WGAN',
            seed=seed,
            discriminator_config=discriminator_config,
            gan_config=gan_config,
            generator_config=generator_config
        )


def benchmark_sigwgan(
        datasets=list_of_datasets,
        generators=('LSTM', 'NSDE',),
        n_seeds=10,
        device='cuda:0',
):
    """ Benchmark for SigWGAN. """
    seeds = list(range(n_seeds))

    grid = itertools.product(datasets, generators, seeds)

    for dataset, generator, seed in grid:
        data_config = load_obj(get_config_path(dataset, dataset))
        gan_config = load_obj(get_config_path('SigWGAN', dataset))
        generator_config = load_obj(get_config_path_generator(generator, dataset))

        if gan_config.get('augmentations') is not None:
            gan_config['augmentations'] = parse_augmentations(gan_config.get('augmentations'))

        if generator_config.get('augmentations') is not None:
            generator_config['augmentations'] = parse_augmentations(generator_config.get('augmentations'))

        if generator_config['generator_type'] == 'LogSigRNN':
            generator_config['n_lags'] = data_config['n_lags']

        experiment_dir = get_sigwgan_experiment_dir(dataset, generator, 'SigWGAN', seed)

        if not pt.exists(experiment_dir):
            os.makedirs(experiment_dir)

        save_obj(data_config, pt.join(experiment_dir, 'data_config.pkl'))
        save_obj(gan_config, pt.join(experiment_dir, 'gan_config.pkl'))
        save_obj(generator_config, pt.join(experiment_dir, 'generator_config.pkl'))

        print('Training: %s' % experiment_dir.split('/')[-2:])

        main(
            dataset=dataset,
            data_config=data_config,
            device=device,
            experiment_dir=experiment_dir,
            gan_algo='SigWGAN',
            seed=seed,
            gan_config=gan_config,
            generator_config=generator_config,
        )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device)
    else:
        device = 'cpu'
    # Test run
    benchmark_sigwgan(datasets=('STOCKS', 'GBM'), generators=('LogSigRNN', 'LSTM',), n_seeds=1, device=device)
    benchmark_wgan(datasets=('STOCKS', 'GBM'), generators=('LogSigRNN', 'LSTM'), n_seeds=1, device=device)
