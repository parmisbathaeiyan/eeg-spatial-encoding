"""Per-electrode-neighborhood ("searchlight") classical classifiers."""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
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


def run_searchlight(X, y, neighbor_idx, ch_names, classifiers=None,
                    n_folds=5, random_state=42, log_every=10):
    """Per-electrode-neighborhood stratified CV.

    X: (N, n_channels, n_features) - for each electrode, the neighborhood's channels are
    selected and flattened to (K+1)*n_features. Returns a (n_channels x n_classifiers)
    DataFrame of mean CV accuracy, indexed by ch_names.
    """
    if classifiers is None:
        classifiers = default_classifiers(random_state)

    n_channels = X.shape[1]
    results = {name: np.zeros(n_channels) for name in classifiers}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for ch in range(n_channels):
        X_ch = X[:, neighbor_idx[ch], :].reshape(X.shape[0], -1)  # (N, (K+1)*n_features)
        for name, make_clf in classifiers.items():
            acc = []
            for train_idx, test_idx in skf.split(X_ch, y):
                pipe = make_pipeline(StandardScaler(), make_clf())
                pipe.fit(X_ch[train_idx], y[train_idx])
                acc.append(pipe.score(X_ch[test_idx], y[test_idx]))
            results[name][ch] = np.mean(acc)
        if log_every and ch % log_every == 0:
            print(f"  channel {ch}/{n_channels} done")

    return pd.DataFrame(results, index=list(ch_names))


def report_searchlight(results_df):
    """Print chance level, mean accuracy per classifier, and the best electrode per classifier."""
    print(f"chance level (3-class): {CHANCE_3CLASS:.3f}")
    print(results_df.mean().to_frame("mean_accuracy_across_electrodes"))
    print("best electrode per classifier:")
    for name in results_df.columns:
        print(f"  {name}: {results_df[name].idxmax()} -> {results_df[name].max():.3f}")
