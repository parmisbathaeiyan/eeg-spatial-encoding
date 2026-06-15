# ZuCo Task1-SR — Spatial Sentiment Decoding

For each of the 105 EEG electrodes, train 3-class sentiment classifiers
(negative / neutral / positive) on band-power features from that electrode's spatial
*neighborhood* (itself + its `k` nearest electrodes on the scalp), then plot accuracy as
a scalp topomap ("accuracy across space"). This is a spatial **searchlight** analysis on
the ZuCo 1.0 Task1-SR reading dataset.

## Layout

```
eeg_spatial/
  config.py       Config dataclass (paths + hyperparameters)
  loaders.py      low-level readers for the ZuCo v5 .mat files
  dataset.py      build feature matrices (sentence-averaged / word-concatenated)
  montage.py      electrode montage + spatial k-NN neighbor graph
  searchlight.py  per-electrode-neighborhood classical classifiers
  cnn.py          optional 1D-CNN searchlight on raw EEG (needs torch)
  plotting.py     accuracy topomaps
data/
  zuco_montage.npz   105 channel labels + 3D coords (bundled, no Drive needed)
notebooks/
  Run_ZuCo_Spatial.ipynb   thin Colab driver (clones this repo, runs the pipeline)
fix_sentiment_labels.py     one-off cleaner for the raw sentiment-label CSV
```

## Data (kept on Google Drive, not in this repo)

- `results*_SR.mat` — 12 subject files, ZuCo 1.0 Task1-SR (MATLAB v5 / scipy format).
- Fixed sentiment-label CSV — produced by `fix_sentiment_labels.py`
  (sentiment_label in {-1, 0, 1}).

The electrode montage **is** bundled here (`data/zuco_montage.npz`), so the notebook only
needs Drive for the `.mat` files and the label CSV.

## Colab usage

Open `notebooks/Run_ZuCo_Spatial.ipynb` in Colab. The first cell clones (or `git pull`s)
this repo and installs `requirements.txt`; later cells mount Drive, set the paths in a
`Config`, and run the pipeline. To pick up code changes, just re-run that first cell.

```python
from eeg_spatial.config import Config
from eeg_spatial import dataset, montage, searchlight, plotting

cfg = Config(sr_data_dir=..., labels_csv=..., montage_npz=...)
label_map = dataset.load_label_map(cfg.labels_csv)
X_band, y, meta = dataset.build_band_dataset(cfg, label_map)
ch_names, ch_pos, mont = montage.load_montage(cfg.montage_npz)
neighbor_idx = montage.build_neighbor_graph(ch_pos, cfg.k_neighbors)
info = montage.make_info(ch_names, mont)

results = searchlight.run_searchlight(X_band, y, neighbor_idx, ch_names,
                                      n_folds=cfg.n_folds, random_state=cfg.random_state)
searchlight.report_searchlight(results)
plotting.plot_accuracy_topomaps(results, info, cfg.k_neighbors)
```

### Feature variants

- **Sentence-averaged** (`build_band_dataset`) — each sentence → one `(105, 8)` vector,
  averaged over its words. The main pipeline.
- **Word-concatenated** (`build_concat_dataset`) — keep every word's `(105, 8)`,
  padded/truncated to a fixed word count. Same `run_searchlight`, larger feature vector.

### Optional CNN

Set `cfg.keep_raw_eeg = True`, rebuild the dataset, then call
`cnn.run_cnn_searchlight(meta["raw_eeg"], y, neighbor_idx, ch_names, device)`. Trains one
CNN per electrode neighborhood — use a GPU runtime.
