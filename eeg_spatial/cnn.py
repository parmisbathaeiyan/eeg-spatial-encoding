"""Optional 1D-CNN searchlight on raw EEG (variable trial length via adaptive pooling).

Trains one CNN per electrode neighborhood with the same stratified CV as the classical
searchlight. This trains n_folds * n_channels separate CNNs, so a GPU is strongly
recommended. Requires the dataset to be built with cfg.keep_raw_eeg = True.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}  # {neg, neu, pos} -> CrossEntropyLoss class ids


class EEGDataset(Dataset):
    def __init__(self, trials, labels):
        self.trials = trials   # list of (n_ch, T) arrays
        self.labels = labels

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.tensor(self.trials[idx], dtype=torch.float32), self.labels[idx]


def collate_pad(batch):
    """Zero-pad a batch of (n_ch, T) trials to the batch's max T."""
    trials, labels = zip(*batch)
    max_t = max(t.shape[1] for t in trials)
    n_ch = trials[0].shape[0]
    padded = torch.zeros(len(trials), n_ch, max_t)
    for i, t in enumerate(trials):
        padded[i, :, :t.shape[1]] = t
    return padded, torch.tensor(labels, dtype=torch.long)


class EEGCNN(nn.Module):
    def __init__(self, n_channels, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),  # collapse variable T -> 1 regardless of input length
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1)  # (B, 64)
        return self.fc(x)


def _train_eval_fold(trials, labels, n_ch, train_idx, test_idx, device, epochs, batch_size):
    tr = EEGDataset([trials[i] for i in train_idx], labels[train_idx])
    te = EEGDataset([trials[i] for i in test_idx], labels[test_idx])
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    model = EEGCNN(n_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def run_cnn_searchlight(raw_eeg, y, neighbor_idx, ch_names, device,
                        epochs=15, batch_size=16, n_folds=5, random_state=42, log_every=5):
    """raw_eeg: list of (n_channels, T) arrays aligned with y (labels in {-1,0,1}).

    Returns a length-n_channels array of mean CV accuracy (assign it to results_df['CNN']).
    """
    assert raw_eeg is not None and len(raw_eeg) == len(y), \
        "raw_eeg missing/misaligned with y - build the dataset with cfg.keep_raw_eeg=True."
    y_idx = np.array([LABEL_TO_IDX[v] for v in y])

    n_channels = len(neighbor_idx)
    acc = np.zeros(n_channels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for ch in range(n_channels):
        nbr = neighbor_idx[ch]
        trials = [t[nbr, :] for t in raw_eeg]  # each (K+1, T)
        fold = [_train_eval_fold(trials, y_idx, len(nbr), tr, te, device, epochs, batch_size)
                for tr, te in skf.split(trials, y_idx)]
        acc[ch] = np.mean(fold)
        if log_every and ch % log_every == 0:
            print(f"  CNN channel {ch}/{n_channels} done (acc {acc[ch]:.3f})")
    return acc
