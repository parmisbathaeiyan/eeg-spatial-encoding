"""TeCo (Persian) loaders — the same pipeline as ZuCo, on TeCo Task-1 data.

TeCo differs from ZuCo in a few ways the rest of the package doesn't care about:
  - the .mat files are MATLAB v7.3 (HDF5), read with h5py (ZuCo's are true v5 / scipy);
  - 126 EEG channels (vs 105);
  - 165 sentences, perfectly balanced 55/55/55;
  - labels join on the *Persian* sentence text (normalize: strip whitespace + drop U+200C);
  - unfixated words store a (2,) uint64 stub instead of a (126,) vector.

This module produces the same `(N, n_channels, n_bands)` X / `y` / sentence `groups` that
`searchlight`, `stats`, and `plotting` already consume - so once you have a TeCo features
file + a 126-channel montage, the whole analysis (run_local, permutation test, ...) works
unchanged.
"""
import glob
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import h5py

from .loaders import DEFAULT_BANDS


def norm_persian(s):
    """Normalize Persian text for label matching: strip whitespace + drop the ZWNJ (U+200C)."""
    return re.sub(r"\s+", "", str(s)).replace("‌", "")


def _h5_str(f, ref):
    """Decode a MATLAB char array (HDF5 object ref) to a Python string."""
    return "".join(chr(c) for c in np.asarray(f[ref]).ravel())


def find_teco_files(task1_root):
    """Per-subject TeCo files under task1_with_total/<subject>/. Prefers .pickle, else .mat."""
    pkls = sorted(glob.glob(os.path.join(task1_root, "*", "task1.pickle")))
    if pkls:
        return pkls
    return sorted(glob.glob(os.path.join(task1_root, "*", "task1.mat")))


# backwards-compatible alias
find_teco_mat_files = find_teco_files


def load_teco_label_map(labels_csv):
    """Return {normalized_persian_text: sentiment_label} from teco_sentiment_labels_task1.csv."""
    df = pd.read_csv(labels_csv)
    return {norm_persian(r["sentence"]): r["sentiment_label"] for _, r in df.iterrows()}


def _word_band_vector(f, word_group, wi, feature_prefix, bands, n_channels):
    """(n_channels x n_bands) band vector for one word; NaN where a band is a stub/missing."""
    vec = np.full((n_channels, len(bands)), np.nan, dtype=np.float32)
    for b, suf in enumerate(bands):
        field = f"{feature_prefix}_{suf}"
        if field not in word_group:
            continue
        v = np.asarray(f[word_group[field][wi, 0]]).ravel()
        if v.size == n_channels:           # real vector; (2,) stub means unfixated -> leave NaN
            vec[:, b] = v
    return vec


def load_teco_subject(path, feature_prefix="TRT", bands=DEFAULT_BANDS, n_channels=126):
    """Yield (sentence_id, persian_text, band_features [n_channels x n_bands]) per sentence.

    Dispatches on file type: `.pickle`/`.pkl` (a dict of trials) or `.mat` (v7.3/HDF5).
    band_features is the mean over the sentence's words (unfixated words give a stub -> NaN,
    ignored by the nanmean). Sentences with no usable word data are skipped.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pickle", ".pkl"):
        yield from _load_subject_pickle(path, feature_prefix, bands, n_channels)
    else:
        yield from _load_subject_mat(path, feature_prefix, bands, n_channels)


def _load_subject_pickle(pkl_path, feature_prefix, bands, n_channels):
    with open(pkl_path, "rb") as fh:
        subj = pickle.load(fh)
    for t in subj:
        trial = subj[t]
        words = trial.get("word") or {}
        if not words:
            continue
        vecs = []
        for w in words.values():
            vec = np.full((n_channels, len(bands)), np.nan, dtype=np.float32)
            for b, suf in enumerate(bands):
                v = np.asarray(w.get(f"{feature_prefix}_{suf}"))
                if v.size == n_channels:           # (2,) stub = unfixated -> leave NaN
                    vec[:, b] = v.ravel()
            vecs.append(vec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat = np.nanmean(np.stack(vecs), axis=0)
        if np.isnan(feat).all():
            continue
        yield int(trial["sentenceId"]), str(trial["persian_sentence"]), feat


def _load_subject_mat(mat_path, feature_prefix, bands, n_channels):
    with h5py.File(mat_path, "r") as f:
        sd = f["main_structure"]
        n_trials = sd["word"].shape[0]
        key0 = f"{feature_prefix}_{bands[0]}"

        for t in range(n_trials):
            wg = f[sd["word"][t, 0]]
            if key0 not in wg:
                continue
            n_words = wg[key0].shape[0]

            wb = np.full((n_words, n_channels, len(bands)), np.nan, dtype=np.float32)
            for wi in range(n_words):
                wb[wi] = _word_band_vector(f, wg, wi, feature_prefix, bands, n_channels)

            with warnings.catch_warnings():           # all-NaN slice -> NaN, handled below
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feat = np.nanmean(wb, axis=0)          # (n_channels, n_bands)
            if np.isnan(feat).all():
                continue

            sid = int(np.asarray(f[sd["id"][t, 0]]).ravel()[0])
            yield sid, _h5_str(f, sd["persian_sentence"][t, 0]), feat


def build_teco_dataset(mat_files, label_map, feature_prefix="TRT", bands=DEFAULT_BANDS,
                       n_channels=126, verbose=True):
    """Sentence-level TeCo dataset across subjects.

    Returns (X [N, n_channels, n_bands], y [N] in {-1,0,1}, meta). meta['texts'] is the
    normalized Persian sentence per trial - use it as `groups` for leakage-free CV and the
    permutation test, exactly like ZuCo's meta['texts'].
    """
    feats, labels, subjects, texts, sids = [], [], [], [], []
    per_subject = {}

    for mat_path in mat_files:
        subj = os.path.basename(os.path.dirname(mat_path))
        seen = matched = 0
        for sid, text, feat in load_teco_subject(mat_path, feature_prefix, bands, n_channels):
            seen += 1
            key = norm_persian(text)
            if key not in label_map:
                continue
            matched += 1
            feats.append(feat)
            labels.append(label_map[key])
            subjects.append(subj)
            texts.append(key)
            sids.append(sid)
        per_subject[subj] = (seen, matched)
        if verbose:
            print(f"  {subj}: {seen} seen, {matched} matched")

    X = np.stack(feats).astype(np.float32)
    y = np.array(labels)

    nan_mask = np.isnan(X).any(axis=(1, 2))
    if nan_mask.any():
        keep = ~nan_mask
        X, y = X[keep], y[keep]
        subjects = [s for s, k in zip(subjects, keep) if k]
        texts = [t for t, k in zip(texts, keep) if k]
        sids = [s for s, k in zip(sids, keep) if k]

    meta = dict(subjects=subjects, texts=texts, sentence_ids=sids,
                per_subject=per_subject, n_dropped_nan=int(nan_mask.sum()))
    if verbose:
        print("X shape:", X.shape, "| dropped NaN trials:", int(nan_mask.sum()))
        print("class distribution:", pd.Series(y).value_counts().to_dict())
    return X, y, meta


def load_teco_montage(location_xyz_txt, rotate_deg=0):
    """Load the 126-channel TeCo montage from location_xyz_reference.txt.

    File format (tab-separated, no header): index, x, y, z, label. Channel order is assumed
    to match the data's channel axis. `rotate_deg` rotates about Z only for topomap
    orientation - it does NOT affect the searchlight (k-NN neighborhoods are distance-based,
    so invariant to rotation), only how the scalp map is drawn.

    Returns (ch_names, ch_pos, montage).
    """
    import mne
    loc = pd.read_csv(location_xyz_txt, sep="\t", header=None,
                      names=["index", "x", "y", "z", "label"])
    ch_names = [str(l) for l in loc["label"]]
    coords = loc[["x", "y", "z"]].to_numpy(dtype=float)

    if rotate_deg:
        th = np.radians(rotate_deg)
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th),  np.cos(th), 0],
                      [0,           0,          1]])
        coords = coords @ R.T

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, coords)), coord_frame="head")
    return ch_names, coords, montage
