"""Topomap plotting of per-electrode decoding accuracy ("accuracy across space")."""
import numpy as np
import matplotlib.pyplot as plt
import mne

CHANCE_3CLASS = 1.0 / 3.0


def plot_accuracy_topomaps(results_df, info, k_neighbors, title_suffix=""):
    """One topomap per column of results_df (mean CV accuracy per electrode)."""
    n_clf = len(results_df.columns)
    n_cols = 3
    n_rows = int(np.ceil(n_clf / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    vmin, vmax = CHANCE_3CLASS - 0.05, results_df.values.max() + 0.02

    im = None
    for ax, clf in zip(axes, results_df.columns):
        im, _ = mne.viz.plot_topomap(
            results_df[clf].values, info, axes=ax, show=False,
            vlim=(vmin, vmax), cmap="RdBu_r", contours=4,
        )
        ax.set_title(f"{clf}\n(mean={results_df[clf].mean():.3f}, chance={CHANCE_3CLASS:.3f})")

    for ax in axes[n_clf:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="CV accuracy")
    plt.suptitle(
        f"3-class sentiment decoding accuracy across electrode neighborhoods "
        f"(k={k_neighbors}){title_suffix}", y=1.02,
    )
    plt.tight_layout()
    plt.show()
