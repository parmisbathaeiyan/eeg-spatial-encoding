"""Per-electrode-neighborhood ("searchlight") classifiers + the shared CV core.

`searchlight_accuracy` is the single per-electrode CV loop reused everywhere (the multi-
classifier `run_searchlight` and the permutation test in `stats.py`). Cross-validation is
built by `make_cv_splits`: by default plain stratified K-fold, or - when `groups` is given -
stratified *grouped* K-fold that holds whole groups (e.g. all of a sentence's trials) out
together, so there is no leakage across the train/test split.
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

CHANCE_3CLASS = 1.0 / 3.0


def default_classifiers(random_state=42):
    """Factory dict {name: () -> fresh estimator}. Each value makes a new instance per fold."""
    return {
        "LogReg": lambda: LogisticRegression(max_iter=1000),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "SVM_RBF": lambda: SVC(kernel="rbf"),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=random_state),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=random_state),
    }


def make_cv_splits(y, groups=None, n_folds=5, random_state=42):
    """Build a fixed list of (train_idx, test_idx) folds.

    groups=None  -> StratifiedKFold (the original setup).
    groups given -> StratifiedGroupKFold: whole groups (e.g. all 12 subjects' copies of a
                    sentence) are kept together in one fold, so the test set is unseen
                    sentences (leakage-free), while classes stay balanced across folds.
    """
    y = np.asarray(y)
    dummy = np.zeros(len(y))
    if groups is None:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return list(cv.split(dummy, y))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(cv.split(dummy, y, np.asarray(groups)))


def searchlight_accuracy(X, y, neighbor_idx, make_clf, splits):
    """Mean CV accuracy per electrode neighborhood, for ONE classifier over fixed folds.

    X: (N, n_channels, n_features); returns a length-n_channels array. This is the shared
    inner loop used by both run_searchlight and the permutation test.
    """
    y = np.asarray(y)
    n_channels = X.shape[1]
    acc = np.zeros(n_channels)
    for ch in range(n_channels):
        X_ch = X[:, neighbor_idx[ch], :].reshape(X.shape[0], -1)  # (N, (K+1)*n_features)
        fold = []
        for train_idx, test_idx in splits:
            pipe = make_pipeline(StandardScaler(), make_clf())
            pipe.fit(X_ch[train_idx], y[train_idx])
            fold.append(pipe.score(X_ch[test_idx], y[test_idx]))
        acc[ch] = np.mean(fold)
    return acc


def run_searchlight(X, y, neighbor_idx, ch_names, classifiers=None,
                    n_folds=5, random_state=42, log_every=1, groups=None):
    """Searchlight CV accuracy per electrode for each classifier.

    Returns a (n_channels x n_classifiers) DataFrame indexed by ch_names. Pass
    `groups=meta['texts']` for leakage-free (sentence-grouped) CV; leave it None to
    reproduce the original stratified-K-fold setup.
    """
    if classifiers is None:
        classifiers = default_classifiers(random_state)

    splits = make_cv_splits(y, groups=groups, n_folds=n_folds, random_state=random_state)

    results = {}
    for i, (name, make_clf) in enumerate(classifiers.items(), 1):
        results[name] = searchlight_accuracy(X, y, neighbor_idx, make_clf, splits)
        if log_every:
            print(f"  [{i}/{len(classifiers)}] {name} done")

    return pd.DataFrame(results, index=list(ch_names))


def report_searchlight(results_df):
    """Print chance level, mean accuracy per classifier, and the best electrode per classifier."""
    print(f"chance level (3-class): {CHANCE_3CLASS:.3f}")
    print(results_df.mean().to_frame("mean_accuracy_across_electrodes"))
    print("best electrode per classifier:")
    for name in results_df.columns:
        print(f"  {name}: {results_df[name].idxmax()} -> {results_df[name].max():.3f}")
