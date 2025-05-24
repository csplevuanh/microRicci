import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def load_data(json_path):
    """
    Load JSON convergence data.
    Returns a dict: { method: [list of residual arrays] }
    """
    with open(json_path, 'r') as f:
        records = json.load(f)

    data = {}
    for rec in records:
        method = rec['method']
        res = np.array(rec['residuals'], dtype=float)
        data.setdefault(method, []).append(res)
    return data


def aggregate_curves(curve_list):
    """
    Given a list of 1D residual arrays (possibly different lengths),
    pad with NaNs to the longest length, then compute per-iteration
    mean and std (ignoring NaNs).
    Returns (iters, mean_res, std_res).
    """
    max_len = max(len(c) for c in curve_list)
    arr = np.full((len(curve_list), max_len), np.nan, dtype=float)
    for i, c in enumerate(curve_list):
        arr[i, :len(c)] = c
    mean_res = np.nanmean(arr, axis=0)
    std_res = np.nanstd(arr, axis=0)
    iters = np.arange(len(mean_res))
    return iters, mean_res, std_res


def plot_convergence(data, output_path, show_std=False):
    """
    Plot convergence curves.
    """
    plt.figure()
    for method, curves in data.items():
        iters, mean_res, std_res = aggregate_curves(curves)
        plt.plot(iters, mean_res, label=method)
        if show_std:
            plt.fill_between(iters,
                             mean_res - std_res,
                             mean_res + std_res,
                             alpha=0.2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Max Residual')
    plt.title('Ricci Flow Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved convergence plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from JSON data."
    )
    parser.add_argument(
        'json_path',
        help="Path to JSON file with convergence records."
    )
    parser.add_argument(
        '--output', '-o', default='convergence.png',
        help="Path to save the plot image."
    )
    parser.add_argument(
        '--show-std', action='store_true',
        help="Shade ±1 std deviation bands."
    )
    args = parser.parse_args()

    data = load_data(args.json_path)
    plot_convergence(data, args.output, show_std=args.show_std)


if __name__ == '__main__':
    main()
