"""Cluster-based permutation test for the spatial searchlight (significance over space).

Leakage-free by construction: cross-validation holds out whole *sentences* (GroupKFold by
sentence, so all 12 subjects' copies of a held-out sentence are in the test fold), and the
null is built by permuting sentiment labels *at the sentence level* (each unique sentence
gets one shuffled label, propagated to its copies). Multiple comparisons across the 105
electrodes are handled with a spatial cluster test (Maris & Oostenveld, 2007) on the
electrode neighbor graph.

Kept separate from `searchlight.py` on purpose - it uses fixed CV folds, a single fast
classifier (LDA), and joblib parallelism, and importing it changes nothing about the
existing pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mne

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed

CHANCE_3CLASS = 1.0 / 3.0


# ----------------------------------------------------------------------------- graph helpers
def _adjacency(neighbor_idx):
    """Symmetric adjacency list from the (n_channels x K+1) neighbor index."""
    n = len(neighbor_idx)
    adj = [set() for _ in range(n)]
    for e, nbrs in enumerate(neighbor_idx):
        for j in nbrs:
            j = int(j)
            if j != e:
                adj[e].add(j)
                adj[j].add(e)
    return adj


def _components(mask, adj):
    """Connected components (as lists of electrode indices) among electrodes where mask is True."""
    seen, comps = set(), []
    for start in np.where(mask)[0]:
        if start in seen:
            continue
        stack, comp = [int(start)], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for v in adj[u]:
                if mask[v] and v not in seen:
                    stack.append(v)
        comps.append(comp)
    return comps


# ----------------------------------------------------------------------------- searchlight core
def _searchlight_acc(X, y, neighbor_idx, make_clf, splits):
    """Mean CV accuracy per electrode neighborhood for a single classifier and fixed folds."""
    n_ch = X.shape[1]
    acc = np.zeros(n_ch)
    for ch in range(n_ch):
        X_ch = X[:, neighbor_idx[ch], :].reshape(X.shape[0], -1)
        fold = []
        for tr, te in splits:
            pipe = make_pipeline(StandardScaler(), make_clf())
            pipe.fit(X_ch[tr], y[tr])
            fold.append(pipe.score(X_ch[te], y[te]))
        acc[ch] = np.mean(fold)
    return acc


def _permute_group_labels(y, groups, rng):
    """Shuffle labels across groups (sentences), keeping each group's copies on one label."""
    uniq, first = np.unique(groups, return_index=True)
    glabels = y[first]                       # one label per unique sentence
    mapping = dict(zip(uniq, glabels[rng.permutation(len(glabels))]))
    return np.array([mapping[g] for g in groups])


# ----------------------------------------------------------------------------- main entry point
def permutation_cluster_searchlight(X, y, neighbor_idx, groups, n_permutations=200,
                                    n_folds=5, cluster_alpha=0.05, random_state=42,
                                    make_clf=None, n_jobs=-1, verbose=True):
    """Significance-over-space via a sentence-level permutation + spatial cluster test.

    Parameters
    ----------
    X : (N, n_channels, n_features)        feature tensor (e.g. X_band)
    y : (N,)                                labels in {-1, 0, 1}
    neighbor_idx : (n_channels, K+1)        electrode neighbor index (defines adjacency)
    groups : (N,)                           sentence id per trial (e.g. meta['texts']) - the CV
                                            grouping AND the permutation unit
    n_permutations : int                    number of label shuffles for the null
    cluster_alpha : float                   per-electrode cluster-forming threshold (one-sided)
    make_clf : callable                     estimator factory; defaults to LDA (fast)
    n_jobs : int                            joblib parallelism (-1 = all cores)

    Returns
    -------
    dict with: accuracy (n_channels,), p_per_electrode (uncorrected), threshold (per-electrode),
    clusters (list of index lists), cluster_pvalues, sig_mask (bool, cluster-corrected),
    null_max_mass, n_permutations.
    """
    if make_clf is None:
        make_clf = lambda: LinearDiscriminantAnalysis()
    groups = np.asarray(groups)

    # fixed folds: hold out whole sentences (no leakage), reused across all permutations so the
    # null differs from the observed only in the label assignment
    splits = list(GroupKFold(n_splits=n_folds).split(np.zeros(len(y)), y, groups))
    adj = _adjacency(neighbor_idx)

    if verbose:
        print(f"observed searchlight (GroupKFold by sentence, {n_folds} folds)...")
    observed = _searchlight_acc(X, y, neighbor_idx, make_clf, splits)

    if verbose:
        print(f"running {n_permutations} sentence-level label permutations (n_jobs={n_jobs})...")

    def _one(seed):
        rng = np.random.default_rng(seed)
        return _searchlight_acc(X, _permute_group_labels(y, groups, rng), neighbor_idx, make_clf, splits)

    null = np.asarray(Parallel(n_jobs=n_jobs)(
        delayed(_one)(random_state + 1 + i) for i in range(n_permutations)))  # (P, n_ch)

    # per-electrode uncorrected p, and a per-electrode cluster-forming threshold from the null
    p_elec = (1 + (null >= observed).sum(axis=0)) / (1 + n_permutations)
    thr = np.quantile(null, 1 - cluster_alpha, axis=0)

    stat = observed - CHANCE_3CLASS
    obs_clusters = _components(observed > thr, adj)
    obs_mass = [float(stat[c].sum()) for c in obs_clusters]

    # null distribution of the largest cluster mass
    null_max = np.zeros(n_permutations)
    for i in range(n_permutations):
        comps = _components(null[i] > thr, adj)
        nstat = null[i] - CHANCE_3CLASS
        null_max[i] = max((nstat[c].sum() for c in comps), default=0.0)

    cluster_p = [(1 + (null_max >= m).sum()) / (1 + n_permutations) for m in obs_mass]

    sig_mask = np.zeros(len(observed), dtype=bool)
    for c, p in zip(obs_clusters, cluster_p):
        if p < cluster_alpha:
            sig_mask[c] = True

    if verbose:
        chance = CHANCE_3CLASS
        print(f"\nchance={chance:.3f} | best electrode acc={observed.max():.3f}")
        print(f"{len(obs_clusters)} candidate cluster(s); "
              f"{int((np.array(cluster_p) < cluster_alpha).sum()) if cluster_p else 0} significant "
              f"at cluster_alpha={cluster_alpha}")
        for c, p, m in sorted(zip(obs_clusters, cluster_p, obs_mass), key=lambda t: t[1]):
            print(f"  cluster size={len(c):2d}  mass={m:.3f}  p={p:.3f}"
                  + ("  *" if p < cluster_alpha else ""))

    return dict(accuracy=observed, p_per_electrode=p_elec, threshold=thr,
                clusters=obs_clusters, cluster_pvalues=cluster_p, sig_mask=sig_mask,
                null_max_mass=null_max, n_permutations=n_permutations)


def plot_significance_topomap(result, info, title="cluster-corrected significance"):
    """Topomap of accuracy with cluster-significant electrodes highlighted (white = chance)."""
    acc, mask = result["accuracy"], result["sig_mask"]
    vmin = min(acc.min(), CHANCE_3CLASS - 1e-3)
    vmax = max(acc.max(), CHANCE_3CLASS + 1e-3)
    norm = TwoSlopeNorm(vcenter=CHANCE_3CLASS, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    mask_params = dict(marker="o", markerfacecolor="k", markeredgecolor="k", markersize=7, linewidth=0)
    im, _ = mne.viz.plot_topomap(
        acc, info, axes=ax, show=False, cmap="RdBu_r", cnorm=norm,
        mask=mask, mask_params=mask_params, contours=0,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"CV accuracy  (white = chance = {CHANCE_3CLASS:.3f})")
    n_sig_clusters = int((np.array(result["cluster_pvalues"]) < 0.05).sum()) if result["cluster_pvalues"] else 0
    ax.set_title(f"{title}\n{int(mask.sum())} sig. electrodes in {n_sig_clusters} cluster(s)")
    plt.show()
