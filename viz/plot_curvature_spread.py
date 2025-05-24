import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_curvature_spread(csv_path, output_path):
    """
    Reads curvature spread CSV and generates a boxplot of curv_std by method.
    """
    df = pd.read_csv(csv_path)
    # Ensure consistent ordering
    methods = df['method'].unique()
    
    # Prepare data for boxplot: list of lists of std values per method
    data = [df[df['method'] == m]['curv_std'].values for m in methods]

    plt.figure()
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel('Curvature Std Dev')
    plt.title('Curvature Spread Across Meshes')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved curvature spread plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot curvature spread boxplots."
    )
    parser.add_argument(
        'csv_path',
        help="Path to curvature_spread.csv"
    )
    parser.add_argument(
        '--output', '-o', default='curvature_spread.png',
        help="Path to save the plot"
    )
    args = parser.parse_args()

    plot_curvature_spread(args.csv_path, args.output)


if __name__ == '__main__':
    main()
