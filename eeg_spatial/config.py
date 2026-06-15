"""Configuration for the ZuCo spatial-decoding pipeline."""
from dataclasses import dataclass
from typing import Tuple

from .loaders import DEFAULT_BANDS


@dataclass
class Config:
    # --- Drive paths (set these in the notebook) ---
    sr_data_dir: str   # folder containing results*_SR.mat (one per subject)
    labels_csv: str    # fixed sentiment-label CSV (from fix_sentiment_labels.py)
    montage_npz: str   # zuco_montage.npz with 'labels' (105,) and 'coords' (105, 3)

    # --- spatial searchlight ---
    k_neighbors: int = 4

    # --- cross-validation ---
    n_folds: int = 5
    random_state: int = 42

    # --- band-power feature ---
    feature_prefix: str = "TRT"            # one of FFD, TRT, GD, GPT
    bands: Tuple[str, ...] = DEFAULT_BANDS

    # --- only needed for the optional CNN section ---
    keep_raw_eeg: bool = False
