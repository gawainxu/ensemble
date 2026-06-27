import numpy as np
import pickle
import os
from scipy import fft
import copy
import hashlib

import torch

##############################
## Spurious_Fourier dataset ##
##############################

def bernoulli(p, size):
    """ Returns a tensor of 1. (True) or 0. (False) resulting from the outcome of a bernoulli random variable of parameter p.

    Args:
        p (float): Parameter p of the Bernoulli distribution
        size (int...): A sequence of integers defining hte shape of the output tensor

    Returns:
        Tensor: Tensor of Bernoulli random variables of parameter p
    """
    return (torch.rand(size) < p).float()


def XOR(a, b):
    """ Returns a XOR b (the 'Exclusive or' gate)

    Args:
        a (bool): First input
        b (bool): Second input

    Returns:
        bool: The output of the XOR gate
    """
    return (a - b).abs()


def get_split(dataset, holdout_fraction, seed=0, sort=False):
    """ Generates the keys that are used to split a Torch TensorDataset into (1-holdout_fraction) / holdout_fraction.

    Args:
        dataset (TensorDataset): TensorDataset to be split
        holdout_fraction (float): Fraction of the dataset that is gonna be in the out (validation) set
        seed (int, optional): seed used for the shuffling of the data before splitting. Defaults to 0.
        sort (bool, optional): If ''True'' the dataset is gonna be sorted after splitting. Defaults to False.

    Returns:
        list: in (1-holdout_fraction) keys of the split
        list: out (holdout_fraction) keys of the split
    """

    split = int(len(dataset) * holdout_fraction)

    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)

    in_keys = keys[split:]
    out_keys = keys[:split]
    if sort:
        in_keys.sort()
        out_keys.sort()

    return in_keys, out_keys


def make_split(dataset, holdout_fraction, seed=0, sort=False):
    """ Split a Torch TensorDataset into (1-holdout_fraction) / holdout_fraction.

    Args:
        dataset (TensorDataset): Tensor dataset that has 2 tensors -> data, targets
        holdout_fraction (float): Fraction of the dataset that is gonna be in the validation set
        seed (int, optional): seed used for the shuffling of the data before splitting. Defaults to 0.
        sort (bool, optional): If ''True'' the dataset is gonna be sorted after splitting. Defaults to False.

    Returns:
        TensorDataset: 1-holdout_fraction part of the split
        TensorDataset: holdout_fractoin part of the split
    """

    in_keys, out_keys = get_split(dataset, holdout_fraction, seed=seed, sort=sort)

    in_split = dataset[in_keys]
    out_split = dataset[out_keys]

    return torch.utils.data.TensorDataset(*in_split), torch.utils.data.TensorDataset(*out_split)


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.

    This is took from DomainBed repository:
        https://github.com/facebookresearch/DomainBed
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def super_sample(signal):
    """
    Sample signals frames with a bunch of offsets
    discarding end data points
    split signal to windows of 50
    stacking
    Args:
        signal (torch.Tensor): Signal to sample

    Returns:
        torch.Tensor: Super sampled signal
    """

    all_signal = torch.zeros(0, 50, 1)
    for i in range(0, 50, 2):
        new_signal = copy.deepcopy(signal)[i:i - 50]
        split_signal = new_signal.reshape(-1, 50, 1).clone().detach().float()
        all_signal = torch.cat((all_signal, split_signal), dim=0)

    return all_signal


class Spurious_Fourier():
    """ Spurious_Fourier dataset

    A dataset of 1D sinusoid signal to classify according to their Fourier spectrum.
    Peaks in the fourier spectrum are added to the signal that are spuriously correlated to the label.
    Different environment have different correlation rates between the labels and the spurious peaks in the spectrum.

    Args:
        training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file

    Note:
        No download is required as it is purely synthetic
    """
    ## Data parameters
    SEQ_LEN = 50
    PRED_TIME = [49]
    INPUT_SHAPE = [1]
    OUTPUT_SIZE = 2

    ## Domain parameters
    #:float: Level of noise added to the labels
    LABEL_NOISE = 0.25
    #:list: The correlation rate between the label and the spurious peaks
    ENVS = [0.1, 0.8, 0.9]
    SWEEP_ENVS = [0]

    def __init__(self, training_hparams):
        super().__init__()

        ## Save stuff
        self.device = training_hparams['device']
        self.holdout_fraction = training_hparams['holdout_fraction']
        self.trial_seed = training_hparams['trial_seed']

        ## Define label 0 and 1 Fourier spectrum
        self.fourier_0 = np.zeros(1000)
        self.fourier_0[900] = 1
        self.fourier_1 = np.zeros(1000)
        self.fourier_1[700] = 1

        ## Define the spurious Fourier spectrum (one direct and the inverse correlation)
        self.direct_fourier_0 = copy.deepcopy(self.fourier_0)
        self.direct_fourier_1 = copy.deepcopy(self.fourier_1)
        self.direct_fourier_0[200] = 0.5
        self.direct_fourier_1[400] = 0.5

        self.inverse_fourier_0 = copy.deepcopy(self.fourier_0)
        self.inverse_fourier_1 = copy.deepcopy(self.fourier_1)
        self.inverse_fourier_0[400] = 0.5
        self.inverse_fourier_1[200] = 0.5

        ## Create the sequences for direct and inverse
        direct_signal_0 = fft.irfft(self.direct_fourier_0, n=10000)
        direct_signal_0 = torch.tensor(direct_signal_0).float()
        direct_signal_0 /= direct_signal_0.max()
        direct_signal_0 = super_sample(direct_signal_0)
        direct_signal_1 = fft.irfft(self.direct_fourier_1, n=10000)
        direct_signal_1 = torch.tensor(direct_signal_1).float()
        direct_signal_1 /= direct_signal_1.max()
        direct_signal_1 = super_sample(direct_signal_1)

        perm_0 = torch.randperm(direct_signal_0.shape[0])
        direct_signal_0 = direct_signal_0[perm_0, :]
        perm_1 = torch.randperm(direct_signal_1.shape[0])
        direct_signal_1 = direct_signal_1[perm_1, :]
        direct_signal = [direct_signal_0, direct_signal_1]

        inverse_signal_0 = fft.irfft(self.inverse_fourier_0, n=10000)
        inverse_signal_0 = torch.tensor(inverse_signal_0).float()
        inverse_signal_0 /= inverse_signal_0.max()
        inverse_signal_0 = super_sample(inverse_signal_0)
        inverse_signal_1 = fft.irfft(self.inverse_fourier_1, n=10000)
        inverse_signal_1 = torch.tensor(inverse_signal_1).float()
        inverse_signal_1 /= inverse_signal_1.max()
        inverse_signal_1 = super_sample(inverse_signal_1)

        perm_0 = torch.randperm(inverse_signal_0.shape[0])
        inverse_signal_0 = inverse_signal_0[perm_0, :]
        perm_1 = torch.randperm(inverse_signal_1.shape[0])
        inverse_signal_1 = inverse_signal_1[perm_1, :]
        inverse_signal = [inverse_signal_0, inverse_signal_1]

        ## Create the environments with different correlations
        env_size = 4000
        self.train_dataset, self.val_dataset = [], []
        for i, e in enumerate(self.ENVS):

            ## Create set of labels
            env_labels_0 = torch.zeros((env_size // 2, 1)).long()
            env_labels_1 = torch.ones((env_size // 2, 1)).long()
            env_labels = torch.cat((env_labels_0, env_labels_1))

            ## Fill signal
            env_signal = torch.zeros((env_size, 50, 1))
            for j, label in enumerate(env_labels):

                # Label noise
                if bool(bernoulli(self.LABEL_NOISE, 1)):
                    # Correlation to label
                    if bool(bernoulli(e, 1)):
                        env_signal[j, ...] = inverse_signal[label][0, ...]
                        inverse_signal[label] = inverse_signal[label][1:, ...]
                    else:
                        env_signal[j, ...] = direct_signal[label][0, ...]
                        direct_signal[label] = direct_signal[label][1:, ...]

                    # Flip the label
                    env_labels[j, -1] = XOR(label, 1)
                else:
                    if bool(bernoulli(e, 1)):
                        env_signal[j, ...] = direct_signal[label][0, ...]
                        direct_signal[label] = direct_signal[label][1:, ...]
                    else:
                        env_signal[j, ...] = inverse_signal[label][0, ...]
                        inverse_signal[label] = inverse_signal[label][1:, ...]

            # Make Tensor dataset
            dataset = torch.utils.data.TensorDataset(env_signal, env_labels)
            in_dataset, out_dataset = make_split(dataset, self.holdout_fraction,
                                                 seed=seed_hash(i, self.trial_seed))
            self.train_dataset.append(in_dataset)
            self.val_dataset.append(out_dataset)

    def run(self):
        return self.train_dataset, self.val_dataset


if __name__ == "__main__":

    training_hparams = {'device': "cpu",
                        "holdout_fraction": 0.3,
                        "trial_seed": 0}
    SF = Spurious_Fourier(training_hparams)
    ts_data_train, ts_data_test = SF.run()

    save_dir = os.path.join(os.getcwd(), "data")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, "ts.plk"), "wb") as f:
        pickle.dump((ts_data_train, ts_data_test), f)