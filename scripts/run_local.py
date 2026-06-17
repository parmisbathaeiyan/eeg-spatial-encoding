"""Run the spatial searchlight + (optional) cluster permutation test locally.

Works off a small prepared-features file (X_band, y, groups) exported from the notebook,
so no .mat files are needed locally. Export it on Colab right after the Build cell:

    import numpy as np
    np.savez('zuco_features.npz', X_band=X_band, y=y, groups=np.asarray(meta['texts']))

Examples (with the venv active):
    python scripts/run_local.py --features zuco_features.npz                       # classical
    python scripts/run_local.py --features zuco_features.npz --grouped             # leakage-free CV
    python scripts/run_local.py --features zuco_features.npz --grouped --permutation --n-perm 100

The permutation test is RESUMABLE: run a batch, let the laptop cool, run another batch and
it accumulates. Each run appends `--n-perm` more permutations to local_results/perm_null.npz
and recomputes the cluster p-values from the full accumulated null. Use --n-jobs to cap
cores (e.g. --n-jobs 4) so it runs cooler.

    python scripts/run_local.py --features zuco_features.npz --grouped --permutation --n-perm 100 --n-jobs 4
    # ...cool off, then repeat the same command to add 100 more (now 200 total)
"""
import argparse
import os
import sys

import numpy as np

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
                    help="run the cluster-based permutation test (needs groups)")
    ap.add_argument("--n-perm", type=int, default=100, help="permutations to ADD this run")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = all cores; lower = cooler)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out-dir", default="local_results", help="where results are saved")
    ap.add_argument("--no-plot", action="store_true", help="skip the matplotlib windows")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.features, allow_pickle=True)
    X_band, y = d["X_band"], d["y"]
    groups = d["groups"] if "groups" in d.files else None

    ch_names, ch_pos, mont = montage.load_montage(args.montage)
    neighbor_idx = montage.build_neighbor_graph(ch_pos, args.k)
    info = montage.make_info(ch_names, mont)
    assert X_band.shape[1] == len(ch_names), "channel-count mismatch between features and montage"

    # --- classical searchlight (always; saved by default) ---
    grp = groups if args.grouped else None
    tag = "grouped" if args.grouped else "ungrouped"
    print(f"classical searchlight ({tag}) ...")
    results = searchlight.run_searchlight(X_band, y, neighbor_idx, ch_names,
                                          n_folds=args.n_folds, random_state=args.random_state, groups=grp)
    searchlight.report_searchlight(results)
    csv = os.path.join(args.out_dir, f"searchlight_{tag}.csv")
    results.to_csv(csv)
    print("saved ->", csv)

    # --- cluster permutation test (resumable, accumulates a null) ---
    if args.permutation:
        if groups is None:
            sys.exit("permutation test needs `groups` in the features file")

        splits = searchlight.make_cv_splits(y, groups=groups, n_folds=args.n_folds,
                                            random_state=args.random_state)
        observed = searchlight.searchlight_accuracy(X_band, y, neighbor_idx, stats.default_clf, splits)

        null_path = os.path.join(args.out_dir, "perm_null.npz")
        if os.path.exists(null_path):
            null = np.load(null_path)["null"]
            n_done = null.shape[0]
            print(f"resuming: {n_done} permutations already on disk")
        else:
            null = np.empty((0, observed.shape[0]))
            n_done = 0

        seeds = [args.random_state + 1 + (n_done + i) for i in range(args.n_perm)]
        print(f"adding {args.n_perm} permutations (total -> {n_done + args.n_perm}), n_jobs={args.n_jobs} ...")
        new = stats.permute_null(X_band, y, neighbor_idx, groups, splits, stats.default_clf,
                                 seeds, n_jobs=args.n_jobs)
        null = np.vstack([null, new]) if null.size else new

        np.savez(null_path, observed=observed, null=null,
                 random_state=args.random_state, n_folds=args.n_folds)
        print(f"saved null ({null.shape[0]} perms) -> {null_path}")

        result = stats.cluster_test(observed, null, neighbor_idx)
        stats.summarize_clusters(result)

        # per-electrode significance table
        import pandas as pd
        sig_csv = os.path.join(args.out_dir, "perm_significance.csv")
        pd.DataFrame({"accuracy": result["accuracy"],
                      "p_value": result["p_per_electrode"],
                      "significant": result["sig_mask"]}, index=ch_names).to_csv(sig_csv)
        print("saved ->", sig_csv)

        if not args.no_plot:
            stats.plot_significance_topomap(result, info)

    if not args.no_plot:
        plotting.plot_accuracy_topomaps(results, info, args.k)


if __name__ == "__main__":
    main()
