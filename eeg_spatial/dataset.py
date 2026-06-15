"""Build feature matrices (sentence-averaged, word-concatenated) from .mat files + labels."""
import glob
import os

import numpy as np
import pandas as pd

from .loaders import load_subject_sentences, load_subject_sentence_words, norm_text


def find_mat_files(sr_data_dir):
    """Sorted list of subject result files (results*_SR.mat) in sr_data_dir."""
    return sorted(glob.glob(os.path.join(sr_data_dir, "*SR.mat")))


def load_label_map(labels_csv):
    """Return {normalized_sentence_text: sentiment_label} from the fixed label CSV."""
    df = pd.read_csv(labels_csv)
    return {norm_text(r["sentence"]): r["sentiment_label"] for _, r in df.iterrows()}


def build_band_dataset(cfg, label_map, verbose=True):
    """Sentence-level dataset: each sentence -> (n_channels x n_bands), averaged over words.

    Returns (X_band [N, n_channels, n_bands], y [N] in {-1,0,1}, meta). meta holds
    subjects / sentence_ids / texts, an optional raw_eeg list (if cfg.keep_raw_eeg),
    and matching diagnostics (n_unmatched, never_matched, per_subject, n_dropped_nan).
    """
    mat_files = find_mat_files(cfg.sr_data_dir)
    feats, labels, subjects, sent_ids, texts, raws = [], [], [], [], [], []
    n_unmatched = 0
    per_subject = {}

    for mat_path in mat_files:
        subj = os.path.basename(mat_path)
        seen = matched = 0
        for idx, text, band, raw in load_subject_sentences(
                mat_path, cfg.feature_prefix, cfg.bands, compute_raw=cfg.keep_raw_eeg):
            seen += 1
            key = norm_text(text)
            if key not in label_map:
                n_unmatched += 1
                continue
            matched += 1
            feats.append(band)
            labels.append(label_map[key])
            subjects.append(subj)
            sent_ids.append(idx)
            texts.append(key)
            if cfg.keep_raw_eeg:
                raws.append(raw)
        per_subject[subj] = (seen, matched)

    X = np.stack(feats).astype(np.float32)
    y = np.array(labels)

    # drop trials with any NaN band feature (sentences with no usable word-level data)
    nan_mask = np.isnan(X).any(axis=(1, 2))
    if nan_mask.any():
        keep = ~nan_mask
        X, y = X[keep], y[keep]
        subjects = [s for s, k in zip(subjects, keep) if k]
        sent_ids = [s for s, k in zip(sent_ids, keep) if k]
        texts = [t for t, k in zip(texts, keep) if k]
        if cfg.keep_raw_eeg:
            raws = [r for r, k in zip(raws, keep) if k]

    never_matched = set(label_map) - set(texts)
    meta = dict(
        subjects=subjects, sentence_ids=sent_ids, texts=texts,
        raw_eeg=raws if cfg.keep_raw_eeg else None,
        n_unmatched=n_unmatched, never_matched=never_matched,
        per_subject=per_subject, n_dropped_nan=int(nan_mask.sum()),
        mat_files=mat_files,
    )

    if verbose:
        print("X_band shape:", X.shape, "| dropped", int(nan_mask.sum()), "NaN trials")
        print("unmatched (no label):", n_unmatched, "| never matched:", len(never_matched))
        print("class distribution:", pd.Series(y).value_counts().to_dict())
        for subj, (seen, matched) in per_subject.items():
            print(f"  {subj}: {seen} seen, {matched} matched")
    return X, y, meta


def build_concat_dataset(cfg, label_map, max_words="p95", verbose=True):
    """Concatenated-word dataset: keep per-word vectors, pad/truncate to a fixed word count.

    Returns (X_concat [N, n_channels, max_words*n_bands], y, max_words_used). Laid out so
    each channel's feature block is [word1's bands, word2's bands, ...], which keeps the
    (N, n_channels, n_features) shape that run_searchlight expects.

    max_words: int, or 'p95' / 'max' to derive it from the word-count distribution.
    """
    mat_files = find_mat_files(cfg.sr_data_dir)
    word_band, labels = [], []
    for mat_path in mat_files:
        for text, wb in load_subject_sentence_words(mat_path, cfg.feature_prefix, cfg.bands):
            key = norm_text(text)
            if key in label_map:
                word_band.append(wb)
                labels.append(label_map[key])

    y = np.array(labels)
    counts = np.array([wb.shape[0] for wb in word_band])

    if max_words == "p95":
        mw = int(np.percentile(counts, 95))
    elif max_words == "max":
        mw = int(counts.max())
    else:
        mw = int(max_words)

    n_bands = len(cfg.bands)
    n_ch = word_band[0].shape[1]
    X = np.zeros((len(word_band), n_ch, mw, n_bands), dtype=np.float32)
    for i, wb in enumerate(word_band):
        w = min(wb.shape[0], mw)
        X[i, :, :w, :] = np.nan_to_num(wb[:w]).transpose(1, 0, 2)  # (n_channels, w, n_bands)
    X = X.reshape(len(word_band), n_ch, mw * n_bands)

    if verbose:
        print(f"sentences: {len(word_band)} | words/sentence: min {counts.min()}, "
              f"median {int(np.median(counts))}, p95 {int(np.percentile(counts, 95))}, "
              f"max {counts.max()}")
        print("MAX_WORDS:", mw, "| X_concat shape:", X.shape,
              "| per-electrode features:", (cfg.k_neighbors + 1) * mw * n_bands)
        print("class distribution:", pd.Series(y).value_counts().to_dict())
    return X, y, mw
