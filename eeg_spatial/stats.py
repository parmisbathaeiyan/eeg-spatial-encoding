"""Cluster-based permutation test for the spatial searchlight (significance over space).

Leakage-free by construction: CV holds out whole *sentences* (grouped folds, so all 12
subjects' copies of a held-out sentence are in the test fold), and the null is built by
permuting sentiment labels *at the sentence level*. Multiple comparisons across the 105
electrodes are handled with a spatial cluster test (Maris & Oostenveld, 2007) on the
electrode neighbor graph.

Reuses the shared searchlight core from `searchlight.py`. The pieces are factored so a long
permutation run can be done in resumable chunks: `permute_null` runs a batch of permutations,
and `cluster_test` computes the corrected result from an accumulated null of any size.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mne

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import Parallel, delayed

from .searchlight import CHANCE_3CLASS, make_cv_splits, searchlight_accuracy


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
    """Connected components (lists of electrode indices) among electrodes where mask is True."""
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


def _permute_group_labels(y, groups, rng):
    """Shuffle labels across groups (sentences), keeping each group's copies on one label."""
    y = np.asarray(y)
    uniq, first = np.unique(groups, return_index=True)
    glabels = y[first]                       # one label per unique sentence
    mapping = dict(zip(uniq, glabels[rng.permutation(len(glabels))]))
    return np.array([mapping[g] for g in groups])


def default_clf():
    """Fast classifier for the permutation test."""
    return LinearDiscriminantAnalysis()


# ----------------------------------------------------------------------------- building blocks
def permute_null(X, y, neighbor_idx, groups, splits, make_clf, seeds, n_jobs=-1, scoring="accuracy"):
    """Searchlight score maps under sentence-level label permutation.

    One map per integer in `seeds` (the seed makes each permutation reproducible). Returns
    an array of shape (len(seeds), n_channels). Use distinct seeds across chunks to keep
    accumulated permutations independent. `scoring` must match the observed map's scoring.
    """
    groups = np.asarray(groups)

    def _one(seed):
        rng = np.random.default_rng(int(seed))
        yp = _permute_group_labels(y, groups, rng)
        return searchlight_accuracy(X, yp, neighbor_idx, make_clf, splits, scoring=scoring)

    return np.asarray(Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds))


def cluster_test(observed, null, neighbor_idx, cluster_alpha=0.05):
    """Spatial cluster test from an observed (n_ch,) map and a null (P, n_ch) array.

    Works for a null of any size, so you can accumulate permutations across runs and call
    this on the combined null. Returns the same dict as `permutation_cluster_searchlight`.
    """
    observed = np.asarray(observed)
    null = np.asarray(null)
    adj = _adjacency(neighbor_idx)
    n_perm = null.shape[0]

    p_elec = (1 + (null >= observed).sum(axis=0)) / (1 + n_perm)
    thr = np.quantile(null, 1 - cluster_alpha, axis=0)

    stat = observed - CHANCE_3CLASS
    obs_clusters = _components(observed > thr, adj)
    obs_mass = [float(stat[c].sum()) for c in obs_clusters]

    null_max = np.zeros(n_perm)
    for i in range(n_perm):
        comps = _components(null[i] > thr, adj)
        nstat = null[i] - CHANCE_3CLASS
        null_max[i] = max((nstat[c].sum() for c in comps), default=0.0)

    cluster_p = [(1 + (null_max >= m).sum()) / (1 + n_perm) for m in obs_mass]

    sig_mask = np.zeros(len(observed), dtype=bool)
    for c, p in zip(obs_clusters, cluster_p):
        if p < cluster_alpha:
            sig_mask[c] = True

    return dict(accuracy=observed, p_per_electrode=p_elec, threshold=thr,
                clusters=obs_clusters, cluster_pvalues=cluster_p, sig_mask=sig_mask,
                null_max_mass=null_max, n_permutations=n_perm)


def summarize_clusters(result, cluster_alpha=0.05):
    """Print chance, best electrode, and each candidate cluster's size/mass/p-value."""
    cp = result["cluster_pvalues"]
    n_sig = int((np.array(cp) < cluster_alpha).sum()) if cp else 0
    print(f"\nchance={CHANCE_3CLASS:.3f} | best electrode acc={result['accuracy'].max():.3f} "
          f"| {result['n_permutations']} permutations")
    print(f"{len(result['clusters'])} candidate cluster(s); {n_sig} significant at "
          f"cluster_alpha={cluster_alpha}")
    for c, p, m in sorted(zip(result["clusters"], cp,
                              [float((result['accuracy'][c] - CHANCE_3CLASS).sum()) for c in result['clusters']]),
                          key=lambda t: t[1]):
        print(f"  cluster size={len(c):2d}  mass={m:.3f}  p={p:.3f}" + ("  *" if p < cluster_alpha else ""))


# ----------------------------------------------------------------------------- one-shot entry point
def permutation_cluster_searchlight(X, y, neighbor_idx, groups, n_permutations=200,
                                    n_folds=5, cluster_alpha=0.05, random_state=42,
                                    make_clf=None, n_jobs=-1, scoring="accuracy", verbose=True):
    """Run the whole test in one go (observed + all permutations + cluster correction).

    `scoring` is 'accuracy' (default) or 'balanced' (balanced accuracy, chance = 1/3).
    For chunked/resumable runs, use `make_cv_splits` + `searchlight_accuracy` + `permute_null`
    + `cluster_test` directly (see scripts/run_local.py).
    """
    if make_clf is None:
        make_clf = default_clf
    groups = np.asarray(groups)

    splits = make_cv_splits(y, groups=groups, n_folds=n_folds, random_state=random_state)
    if verbose:
        print(f"observed searchlight (grouped {n_folds}-fold by sentence, scoring={scoring})...")
    observed = searchlight_accuracy(X, y, neighbor_idx, make_clf, splits, scoring=scoring)

    if verbose:
        print(f"running {n_permutations} sentence-level label permutations (n_jobs={n_jobs})...")
    seeds = [random_state + 1 + i for i in range(n_permutations)]
    null = permute_null(X, y, neighbor_idx, groups, splits, make_clf, seeds, n_jobs=n_jobs, scoring=scoring)

    result = cluster_test(observed, null, neighbor_idx, cluster_alpha=cluster_alpha)
    if verbose:
        summarize_clusters(result, cluster_alpha)
    return result


def plot_significance_topomap(result, info, title="cluster-corrected significance"):
    """Topomap of accuracy with cluster-significant electrodes ringed (white = chance)."""
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
