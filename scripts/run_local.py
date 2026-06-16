"""Run the spatial searchlight + (optional) cluster permutation test locally.

Works off a small prepared-features file (X_band, y, groups) exported from the notebook,
so no .mat files are needed locally. Export it on Colab right after the Build cell:

    import numpy as np
    np.savez('zuco_features.npz', X_band=X_band, y=y, groups=np.asarray(meta['texts']))

download `zuco_features.npz` into the repo folder, then (with the venv active) e.g.:

    python scripts/run_local.py --features zuco_features.npz                # classical, ungrouped
    python scripts/run_local.py --features zuco_features.npz --grouped      # leakage-free CV
    python scripts/run_local.py --features zuco_features.npz --grouped --permutation --n-perm 500
"""
import argparse
import os
import sys

import numpy as np

# make `eeg_spatial` importable no matter where this is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_spatial import montage, searchlight, plotting, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="npz with X_band, y, groups")
    ap.add_argument("--montage", default="zuco_montage.npz", help="montage npz (default: repo root)")
    ap.add_argument("--k", type=int, default=4, help="k nearest-neighbor electrodes")
    ap.add_argument("--grouped", action="store_true",
                    help="leakage-free StratifiedGroupKFold by sentence (uses groups)")
    ap.add_argument("--permutation", action="store_true",
                    help="also run the cluster-based permutation test (needs groups)")
    ap.add_argument("--n-perm", type=int, default=500, help="permutations for the cluster test")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional CSV path for the classical results")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X_band, y = d["X_band"], d["y"]
    groups = d["groups"] if "groups" in d.files else None

    ch_names, ch_pos, mont = montage.load_montage(args.montage)
    neighbor_idx = montage.build_neighbor_graph(ch_pos, args.k)
    info = montage.make_info(ch_names, mont)
    assert X_band.shape[1] == len(ch_names), "channel-count mismatch between features and montage"

    grp = groups if args.grouped else None
    print(f"classical searchlight (grouped={bool(args.grouped)}) ...")
    results = searchlight.run_searchlight(X_band, y, neighbor_idx, ch_names,
                                          n_folds=args.n_folds, groups=grp)
    searchlight.report_searchlight(results)
    if args.out:
        results.to_csv(args.out)
        print("saved ->", args.out)

    if args.permutation:
        if groups is None:
            sys.exit("permutation test needs `groups` in the features file")
        perm = stats.permutation_cluster_searchlight(
            X_band, y, neighbor_idx, groups=groups,
            n_permutations=args.n_perm, n_folds=args.n_folds, n_jobs=-1)
        stats.plot_significance_topomap(perm, info)

    plotting.plot_accuracy_topomaps(results, info, args.k)


if __name__ == "__main__":
    main()
