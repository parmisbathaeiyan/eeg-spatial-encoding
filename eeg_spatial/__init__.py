"""ZuCo Task1-SR spatial sentiment-decoding pipeline.

Submodules:
    config      - Config dataclass (paths + hyperparameters)
    loaders     - low-level readers for the ZuCo v5 .mat files
    dataset     - build feature matrices (sentence-averaged / word-concatenated)
    montage     - electrode montage + spatial k-NN neighbor graph
    searchlight - per-electrode-neighborhood classical classifiers
    cnn         - optional 1D-CNN searchlight on raw EEG (needs torch)
    plotting    - accuracy topomaps

Import submodules explicitly, e.g. `from eeg_spatial import dataset, searchlight`.
`cnn` and `plotting` pull in torch / mne respectively, so they are not imported here.
"""

__version__ = "0.1.0"
