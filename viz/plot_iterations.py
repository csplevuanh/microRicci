import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_iterations(csv_path, output_path):
    """
    Reads iteration counts CSV and generates a boxplot of iterations by method.
    """
    df = pd.read_csv(csv_path)
    methods = df['method'].unique()
    
    # Prepare data for boxplot: list of iteration arrays per method
    data = [df[df['method'] == m]['iterations'].values for m in methods]

    plt.figure()
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel('Number of Iterations')
    plt.title('Iteration Counts Across Meshes')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved iterations boxplot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot iteration count boxplots for Ricci flow methods."
    )
    parser.add_argument(
        'csv_path',
        help="Path to CSV with columns method, mesh, iterations, time_sec"
    )
    parser.add_argument(
        '--output', '-o', default='iterations_boxplot.png',
        help="Path to save the boxplot image"
    )
    args = parser.parse_args()

    plot_iterations(args.csv_path, args.output)


if __name__ == '__main__':
    main()
