import os
from os import path as pt
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch
import wfdb
from tqdm import tqdm
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import torch

from lib.utils import sample_indices
from fbm import fbm, MBM

DATA_DIR = 'datasets'


class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x):
        x = x.clone()
        for step in self.steps:
            x = step.transform(x)
        return x

    def inverse_transform(self, x):
        for step in self.steps[::-1]:
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(0, 1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


class LogTransform():

    def transform(self, x):
        return x.log()

    def inverse_transform(self, x):
        return x.exp()


class ReturnTransform():

    def transform(self, x):
        return x[:, 1:] - x[:, :-1]

    def inverse_transform(self, x):
        initial_point = torch.zeros(x.shape[0], 1, x.shape[-1]).to(x.device)
        return torch.cat([initial_point, x.cumsum(1)], dim=1)


def get_mit_arrythmia_dataset(datadir, id):
    """ Get preprocessed ECG data for patient ID. """
    ecg_data_loc = './{datadir}/mit-bih-arrhythmia-database-1.0.0/'.format(datadir=datadir)
    records = list()
    records.append(wfdb.rdsamp(os.path.join(ecg_data_loc, str(id)), sampto=3000)[0][None, ...])
    records = np.concatenate(records, axis=0)
    records = np.log(5 * (records - records.min() + 1))
    x = torch.from_numpy(records).float()
    return x


def get_stocks_dataset(
        assets,
        datadir,
        start: str = '2005-01-01 00:00:00+01:00',
        end: str = '2020-06-01 00:00:00+01:00',
        vol_type: str = 'medrv',

):
    """ Get preprocessed stocks data including realized vol + log-returns. """
    oxford = pd.read_csv(
        './{datadir}/oxfordmanrealizedvolatilityindices.csv'.format(datadir=datadir)
    )

    # collect ticker dfs
    dfs = list()
    for ticker in assets:
        dfs.append(
            oxford[oxford['Symbol'] == ticker].set_index(['Unnamed: 0'])[start:end]
        )

    # find index intersections between ticker dfs
    index = dfs[0].index
    for df, symbol in zip(dfs, assets):
        index = index.intersection(df.index)

    vols = list()
    logrtns = list()
    for df in dfs:
        spot = df.loc[index][['close_price']].values
        logrtn = np.log(spot[1:]) - np.log(spot[:-1])
        logrtns.append(logrtn)
        vols.append(df.loc[index][[vol_type]].values)

    logrtns = np.concatenate(logrtns, axis=1)

    vols = np.concatenate(vols, axis=1)
    vols[vols <= 1e-6] = 1e-6
    vols = np.log(vols)

    x = np.concatenate([logrtns, vols[1:]], axis=1)
    x = torch.from_numpy(x).float().unsqueeze(0)
    return x[..., :1] # we only learn the logreturn


def download_man_ahl_dataset(datadir):
    url = 'https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'
    r = requests.get(url)
    with open('./oxford.zip', 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./oxford.zip')
    zf.extractall(path='./{datadir}'.format(datadir=datadir))
    zf.close()
    os.remove('./oxford.zip')


def download_mit_ecg_dataset(datadir):
    url = 'https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip'
    r = requests.get(url)
    path_to_dataloc = pt.join(datadir, 'mit_bd.zip')
    with open(path_to_dataloc, 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile(path_to_dataloc)
    zf.extractall(path=datadir)
    zf.close()
    os.remove(path_to_dataloc)


def download_beijing_air_quality_dataset(datadir):
    import requests
    from zipfile import ZipFile
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip'
    r = requests.get(url)
    with open(os.path.join(datadir, './beijing.zip'), 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile(os.path.join(datadir, 'beijing.zip'))
    zf.extractall(path=datadir)
    zf.close()
    os.remove(os.path.join(datadir, 'beijing.zip'))


def get_data_beijing_air_quality(datadir):
    """
    Get Beijin air quality dataset
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (batch_size, 24, 6)
    """
    files = glob.glob(os.path.join(datadir, 'PRSA_Data_20130301-20170228/*csv'))
    df = pd.concat([pd.read_csv(f) for f in files], 0)
    columns_interest = ['SO2', 'NO2', 'CO', 'O3', 'PM2.5', 'PM10']
    dataset = []
    for idx, (ind, group) in enumerate(df.groupby(['year', 'month', 'day', 'station'])):
        dataset.append(group[columns_interest].values)
    dataset = np.stack(dataset, axis=0)
    return torch.from_numpy(dataset).float()


def rolling_window(x: torch.Tensor, n_lags):
    return torch.cat([x[:, t:t + n_lags] for t in range(x.shape[1] - n_lags + 1)], dim=0)


def get_dataset(dataset: str, data_config: dict, n_lags: int, datadir=DATA_DIR):
    """
    Loads different datasets and downloads missing datasets.

    Parameters
    ----------
    dataset: str, specifies which dataset loading function to use
    data_config: dict, contains kwargs for loading the dataset
    n_lags: int, length of (rolled) paths
    Returns
    -------

    x_real: torch.Tensor, dataset

    """
    if dataset == 'GBM':

        def get_gbm(size, n_lags, d=1, drift=0., scale=0.1, h=1):
            x_real = torch.ones(size, n_lags, d)
            x_real[:,1:,:] = torch.exp((drift-scale**2/2)*h + (scale*np.sqrt(h)*torch.randn(size,n_lags-1,d)))
            x_real = x_real.cumprod(1)
            return x_real

        x_real = get_gbm(n_lags=n_lags+1, **data_config)
    elif dataset == 'ROUGH' or dataset == 'ROUGH_S':
        path_rough = pt.join(DATA_DIR, 'rBergomi_{}steps.pth.tar'.format(n_lags))
        if pt.exists(path_rough):
            x_real = torch.load(path_rough, map_location="cpu")
            x_real = x_real[..., :1] if dataset == 'ROUGH_S' else x_real
        else:
            x_real = get_rBergomi_paths(n_lags=n_lags,**data_config)
            x_real = torch.from_numpy(x_real)
            torch.save(x_real, path_rough)
            x_real = x_real[..., :1] if dataset == 'ROUGH_S' else x_real
    elif dataset == 'STOCKS':
        if not pt.exists(
                pt.join(DATA_DIR, 'oxfordmanrealizedvolatilityindices.csv')
        ):
            print('Downloading Oxford MAN AHL realised library.')
            download_man_ahl_dataset(datadir)
        x_real = get_stocks_dataset(datadir=datadir, **data_config)
        x_real = rolling_window(x_real, n_lags)
    elif dataset == 'ECG':
        if not pt.exists(
                pt.join(DATA_DIR, 'mit-bih-arrhythmia-database-1.0.0')
        ):
            print('Downloading MIT-ECG database.')
            download_mit_ecg_dataset(datadir)
        x_real = get_mit_arrythmia_dataset(datadir=datadir, **data_config)
        pipeline = Pipeline([LogTransform(), StandardScalerTS()])
        x_real = pipeline.transform(x_real)
    elif dataset == 'BEIJING':
        if not pt.exists(pt.join(datadir, 'PRSA_Data_20130301-20170228')):
            print('Downloading Bejing air quality dataset.')
            download_beijing_air_quality_dataset(datadir)
        x_real = get_data_beijing_air_quality(datadir)
    else:
        raise NotImplementedError('Dataset %s not valid' % dataset)
    assert len(x_real.shape) == 3
    return x_real.float()


def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float
):
    """
    Apply a train-test split to a given tensor along the first dimension of the tensor.

    Parameters
    ----------
    x: torch.Tensor, tensor to split.
    train_test_ratio, percentage of samples kept in train set, i.e. 0.8 => keep 80% of samples in the train set

    Returns
    -------
    x_train: torch.Tensor, training set
    x_test: torch.Tensor, test set
    """
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size)
    indices_test = torch.LongTensor([i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test



# Functions to simulate fbm and rough volatility model
def ComputeY(Y_last, dt, dB, step):
    ans = Y_last + (-np.pi * Y_last + np.sin(np.pi * step * dt)) * dt + Y_last * dB + 0.5 * Y_last * (dB * dB - dt)
    return ans


def get_rBergomi_paths(hurst=0.25, size=2200, n_lags=100, maturity=1, xi=0.5, eta=0.5):
    r"""
    Paths of Rough stochastic volatility model for an asset price process S_t of the form

    dS_t = \sqrt(V_t) S_t dZ_t
    V_t := \xi * exp(\eta * W_t^H - 0.5*\eta^2*t^{2H})

    where W_t^H denotes the Riemann-Liouville fBM given by

    W_t^H := \int_0^t K(t-s) dW_t,  K(r) := \sqrt{2H} r^{H-1/2}

    with W_t,Z_t correlated brownian motions (I'm actually considering \rho=0)

    Parameters
    ----------
    hurst: float,
    size: int
        size of the dataset
    n_lags: int
        Number of timesteps in the path
    maturity: float
        Final time. Should be a value in [0,1]
    xi: float
    eta: float

    Returns
    -------
    dataset: np.array
        array of shape (size, n_lags, 2)

    """
    assert hurst<0.5, "hurst parameter should be < 0.5"

    dataset = np.zeros((size, n_lags, 2))

    for j in tqdm(range(size), total=size):
        # we generate v process
        m = MBM(n=n_lags-1, hurst=lambda t: hurst, length=maturity, method='riemannliouville')
        fbm = m.mbm() # fractional Brownian motion
        times = m.times()
        V = xi * np.exp(eta * fbm - 0.5 * eta**2 * times**(2*hurst))

        # we generate price process
        h = times[1:] - times[:-1] # time increments
        brownian_increments = np.random.randn(h.shape[0]) * np.sqrt(h)

        log_S = np.zeros_like(V)
        log_S[1:] = (-0.5 * V[:-1]*h + np.sqrt(V[:-1]) * brownian_increments).cumsum() # Ito formula to get SDE for  d log(S_t). We assume S_0 = 1
        S = np.exp(log_S)
        dataset[j] = np.stack([S, V],1) 
    return dataset

