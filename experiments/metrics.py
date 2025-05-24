import os
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_curvature_spread(curv_dir, out_csv):
    """
    Expects .npz files named '<method>_<mesh>_curv.npz' each containing
    an array 'curvature' of shape (n_vertices,).
    Writes a CSV with columns: method, mesh, curv_mean, curv_std.
    """
    records = []
    for fname in os.listdir(curv_dir):
        if not fname.endswith('.npz'):
            continue
        try:
            base = fname[:-4]
            method, mesh, _ = base.split('_', 2)
            data = np.load(os.path.join(curv_dir, fname))
            curv = data['curvature']
            records.append({
                'method': method,
                'mesh': mesh,
                'curv_mean': float(np.mean(curv)),
                'curv_std': float(np.std(curv))
            })
        except Exception as e:
            print(f"Warning: failed to process {fname}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved curvature spread to {out_csv} ({len(df)} entries).")
    return df


def compute_uv_mos_correlation(uv_mos_csv, out_json):
    """
    Expects a CSV with columns: mesh, uv_error (float), mos_score (float).
    Computes Pearson & Spearman correlations between uv_error and mos_score.
    Writes results to a JSON file.
    """
    df = pd.read_csv(uv_mos_csv)
    # drop NaNs
    df = df.dropna(subset=['uv_error', 'mos_score'])
    x = df['uv_error'].values
    y = df['mos_score'].values

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    results = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_pairs': int(len(df))
    }
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved UV–MOS correlations to {out_json}.")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute curvature spread and UV–MOS correlations."
    )
    parser.add_argument(
        '--curv-dir', required=True,
        help='Directory containing <method>_<mesh>_curv.npz files.'
    )
    parser.add_argument(
        '--uv-mos-csv', required=True,
        help='CSV of UV error vs. MOS: columns mesh, uv_error, mos_score.'
    )
    parser.add_argument(
        '--out-dir', default='results/metrics',
        help='Directory to write metrics outputs.'
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    curv_csv = os.path.join(args.out_dir, 'curvature_spread.csv')
    uvmos_json = os.path.join(args.out_dir, 'uv_mos_correlation.json')

    compute_curvature_spread(args.curv_dir, curv_csv)
    compute_uv_mos_correlation(args.uv_mos_csv, uvmos_json)


if __name__ == '__main__':
    main()
