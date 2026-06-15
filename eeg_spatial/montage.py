"""Electrode montage + spatial k-NN neighbor graph."""
import numpy as np
import mne
from sklearn.neighbors import NearestNeighbors


def load_montage(montage_npz, rotate_deg=90):
    """Load ZuCo electrode labels + 3D positions, rotated into MNE's 'head' frame.

    The raw ZuCo coords have +X toward the nose; MNE's head frame uses +Y = anterior,
    +X = right ear, so we rotate `rotate_deg` degrees CCW about Z (same correction used
    in Interpolate_both_datasets.ipynb).

    Returns (ch_names [list of str], ch_pos [n x 3], montage [mne DigMontage]).
    """
    data = np.load(montage_npz, allow_pickle=True)
    ch_names = [str(c) for c in data["labels"]]
    coords = data["coords"]  # (n, 3), in cm, from ZuCo chanlocs

    theta = np.radians(rotate_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    ch_pos = coords @ R.T

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, ch_pos)), coord_frame="head")
    return ch_names, ch_pos, montage


def build_neighbor_graph(ch_pos, k_neighbors):
    """neighbor_idx (n_channels x (k+1)); column 0 is the electrode itself."""
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(ch_pos)
    _, neighbor_idx = nbrs.kneighbors(ch_pos)
    return neighbor_idx


def make_info(ch_names, montage, sfreq=500.0):
    """An mne.Info with the montage set, for topomap plotting."""
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)
    return info
