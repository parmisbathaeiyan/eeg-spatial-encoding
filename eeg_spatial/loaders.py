"""Low-level readers for ZuCo Task1-SR .mat files (MATLAB v5 / scipy format).

These files are *not* v7.3/HDF5 - they load with scipy.io.loadmat. Each file holds a
`sentenceData` struct array; each sentence has a `word` struct array, and each word
carries band-power fields like `TRT_t1`..`TRT_g2` (105-vector per band) and `rawEEG`.
"""
import numpy as np
import scipy.io as sio

# theta1/2, alpha1/2, beta1/2, gamma1/2
DEFAULT_BANDS = ("t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2")


def norm_text(s):
    """Normalize sentence text for matching .mat content against the label CSV."""
    return " ".join(str(s).split()).strip()


def _word_list(word_field):
    """Normalize sentenceData[i].word into a list of word structs (possibly empty)."""
    if isinstance(word_field, np.ndarray):
        if word_field.size == 0:
            return []
        return list(np.atleast_1d(word_field))
    if hasattr(word_field, "_fieldnames"):
        return [word_field]
    return []  # NaN / empty scalar -> no word-level data


def band_vector(word, feature_prefix, bands=DEFAULT_BANDS, n_channels=105):
    """(n_channels x n_bands) band-power vector for one word; NaN where a band is missing."""
    vec = np.full((n_channels, len(bands)), np.nan, dtype=np.float32)
    for b, suf in enumerate(bands):
        field = f"{feature_prefix}_{suf}"
        if not hasattr(word, field):
            continue
        val = np.asarray(getattr(word, field), dtype=np.float32).flatten()
        if val.size == n_channels:
            vec[:, b] = val
    return vec


def raw_eeg(word, raw_field="rawEEG", n_channels=105):
    """Concatenate a word's raw EEG fixation chunks -> (n_channels, T), or None."""
    if not hasattr(word, raw_field):
        return None
    raw = getattr(word, raw_field)

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        chunks = [np.asarray(c, dtype=float) for c in raw.flatten()]
    else:
        arr = np.asarray(raw, dtype=float)
        chunks = [arr] if arr.size > 0 else []

    chunks = [c for c in chunks if c.ndim == 2 and c.size > 0]
    chunks = [c for c in chunks if n_channels in c.shape]
    chunks = [c.T if c.shape[1] == n_channels else c for c in chunks]  # -> (n_channels, T_word)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=1)


def load_subject_sentences(mat_path, feature_prefix="TRT", bands=DEFAULT_BANDS,
                           n_channels=105, compute_raw=False):
    """Yield (sentence_idx, text, band_features [n_channels x n_bands], raw_eeg [n_channels x T]).

    band_features is the mean over the sentence's words. raw_eeg is None unless
    compute_raw=True. Sentences with no usable word-level data are skipped.
    """
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    sd = np.atleast_1d(mat["sentenceData"])

    for s, sent in enumerate(sd):
        words = _word_list(sent.word)
        if not words:
            continue

        band_vals = np.full((len(words), n_channels, len(bands)), np.nan, dtype=np.float32)
        raw_chunks = []
        for w, word in enumerate(words):
            band_vals[w] = band_vector(word, feature_prefix, bands, n_channels)
            if compute_raw:
                r = raw_eeg(word, n_channels=n_channels)
                if r is not None:
                    raw_chunks.append(r)

        band_features = np.nanmean(band_vals, axis=0)  # (n_channels, n_bands)
        raw = np.concatenate(raw_chunks, axis=1) if raw_chunks else None

        if np.isnan(band_features).all():
            continue
        yield s, str(sent.content), band_features, raw


def load_subject_sentence_words(mat_path, feature_prefix="TRT", bands=DEFAULT_BANDS, n_channels=105):
    """Yield (text, word_band [n_words x n_channels x n_bands]) per sentence (no averaging)."""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    for sent in np.atleast_1d(mat["sentenceData"]):
        words = _word_list(sent.word)
        if not words:
            continue
        wb = np.stack([band_vector(w, feature_prefix, bands, n_channels) for w in words])
        if np.isnan(wb).all():
            continue
        yield str(sent.content), wb
