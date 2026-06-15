"""Topomap plotting of per-electrode decoding accuracy ("accuracy across space")."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mne

CHANCE_3CLASS = 1.0 / 3.0


def plot_accuracy_topomaps(results_df, info, k_neighbors, title_suffix="", contours=4):
    """One topomap per column of results_df (mean CV accuracy per electrode).

    The diverging colormap is centered on chance: white == chance, red = above, blue =
    below, so colour is directly interpretable. A single shared colorbar sits to the right
    and `constrained_layout` keeps the two-line titles and colorbar from overlapping.
    """
    n_clf = len(results_df.columns)
    n_cols = min(3, n_clf)
    n_rows = int(np.ceil(n_clf / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.9 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)

    # center the diverging colormap on chance so white == chance level
    vmin = min(results_df.values.min(), CHANCE_3CLASS - 1e-3)
    vmax = max(results_df.values.max(), CHANCE_3CLASS + 1e-3)
    norm = TwoSlopeNorm(vcenter=CHANCE_3CLASS, vmin=vmin, vmax=vmax)

    im = None
    for ax, clf in zip(axes, results_df.columns):
        im, _ = mne.viz.plot_topomap(
            results_df[clf].values, info, axes=ax, show=False,
            cmap="RdBu_r", cnorm=norm, contours=contours,
        )
        ax.set_title(f"{clf}\nmean = {results_df[clf].mean():.3f}", fontsize=11)

    # hide any unused panels (e.g. the 6th slot when there are 5 classifiers)
    for ax in axes[n_clf:]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"CV accuracy  (white = chance = {CHANCE_3CLASS:.3f})")

    fig.suptitle(
        f"3-class sentiment decoding accuracy across electrode neighborhoods "
        f"(k={k_neighbors}){title_suffix}", fontsize=13,
    )
    plt.show()
