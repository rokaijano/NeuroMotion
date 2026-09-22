import torch
import numpy as np
import matplotlib.pyplot as plt

from BioMime.utils.params import coeff_a, coeff_b, w_amp


def generate_emg_mu(muaps, spikes, time_samples):
    """
    Args:
        muaps (np.array): [time_steps, nrow, ncol, duration]
        spikes (list): indices of spikes
        time_samples (int): fs * movement_time

    Return:
        EMG (np.array): [nrow, ncol, time_samples]
    """

    muap_steps, nrow, ncol, time_length = muaps.shape
    emg = np.zeros((nrow, ncol, time_samples + time_length))
    for t in spikes:
        muap_time_id = get_cur_muap(muap_steps, t, time_samples)
        emg[:, :, t:t + time_length] += muaps[muap_time_id]

    return emg


def normalise_physical(values, label):
    """Map BioMime physical parameters into its affine condition space."""
    return (values + coeff_a[label]) * coeff_b[label]


def normalise_properties(db, num_mus, steps=1):

    # Keep values physical until movement factors have been applied.
    num = torch.from_numpy(db['num_fibre_log']).reshape(num_mus, 1).repeat(1, steps)
    depth = torch.from_numpy(db['mu_depth']).reshape(num_mus, 1).repeat(1, steps)
    angle = torch.from_numpy(db['mu_angle']).reshape(num_mus, 1).repeat(1, steps)
    iz = torch.from_numpy(db['iz']).reshape(num_mus, 1).repeat(1, steps)
    cv = torch.from_numpy(db['velocity']).reshape(num_mus, 1).repeat(1, steps)
    length = torch.from_numpy(db['len']).reshape(num_mus, 1).repeat(1, steps)

    base_muap = db['muap'].transpose(0, 3, 1, 2) * w_amp
    base_muap = torch.from_numpy(base_muap).unsqueeze(1).float()

    return num, depth, angle, iz, cv, length, base_muap


def get_cur_muap(muap_steps, cur_step, time_samples):
    return int(muap_steps * cur_step / time_samples)


def plot_spike_trains(spikes, pth):
    """
    spikes  list (index of spikes) in list (MUs)
    pth     figure save path
    """

    fig = plt.figure()
    num_mu = len(spikes)
    for mu in range(num_mu):
        spike = spikes[mu]
        plt.vlines(spike, mu, mu + 0.5, linewidth=1.0)
    plt.xlabel('Time in sample')
    plt.ylabel('MU Index')
    plt.savefig(pth)
