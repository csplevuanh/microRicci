import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def plot_uv_mos(csv_path, output_path):
    """
    Reads UV vs. MOS CSV, plots scatter and best-fit line.
    """
    df = pd.read_csv(csv_path).dropna(subset=['uv_error', 'mos_score'])
    x = df['uv_error'].values
    y = df['mos_score'].values

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept

    plt.figure()
    plt.scatter(x, y, alpha=0.7, label='Data points')
    plt.plot(line_x, line_y, linestyle='--',
             label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\n'
                   f'Pearson r = {r_value:.3f}')
    plt.xlabel('UV Distortion Error')
    plt.ylabel('MOS Score')
    plt.title('UV Distortion vs. MOS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved UV–MOS scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot UV distortion vs. MOS scatter with fit."
    )
    parser.add_argument(
        'csv_path',
        help="Path to CSV with columns mesh, uv_error, mos_score"
    )
    parser.add_argument(
        '--output', '-o', default='uv_mos_scatter.png',
        help="Path to save the scatter plot image"
    )
    args = parser.parse_args()

    plot_uv_mos(args.csv_path, args.output)


if __name__ == '__main__':
    main()
