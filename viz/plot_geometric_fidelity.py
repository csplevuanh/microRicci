import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_boxplot(df, metric, output_path):
    """
    Generate a boxplot for a given metric across methods.
    """
    methods = df['method'].unique()
    data = [df[df['method'] == m][metric].values for m in methods]

    plt.figure()
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Across Meshes')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {metric} plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot geometric fidelity errors (angular & area)."
    )
    parser.add_argument(
        'csv_path',
        help="Path to CSV with columns method, mesh, ang_error, area_error"
    )
    parser.add_argument(
        '--output-prefix', '-o', default='geometric_fidelity',
        help="Prefix for output files (will append _angular.png and _area.png)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Angular error boxplot
    ang_path = f"{args.output_prefix}_angular.png"
    plot_boxplot(df, 'ang_error', ang_path)

    # Area‐ratio error boxplot
    area_path = f"{args.output_prefix}_area.png"
    plot_boxplot(df, 'area_error', area_path)


if __name__ == '__main__':
    main()
