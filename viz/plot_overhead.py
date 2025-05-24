import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_overhead(csv_path, output_path):
    """
    Reads iteration and time CSV, computes per-iteration runtime,
    and generates a boxplot by method.
    """
    df = pd.read_csv(csv_path)
    # Avoid division by zero
    df = df[df['iterations'] > 0].copy()
    df['time_per_iter'] = df['time_sec'] / df['iterations']

    methods = df['method'].unique()
    data = [df[df['method'] == m]['time_per_iter'].values for m in methods]

    plt.figure()
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel('Time per Iteration (s)')
    plt.title('Per-Iteration Runtime Overhead')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved runtime overhead plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-iteration runtime overhead for Ricci flow methods."
    )
    parser.add_argument(
        'csv_path',
        help="Path to CSV with columns method, mesh, iterations, time_sec"
    )
    parser.add_argument(
        '--output', '-o', default='runtime_overhead.png',
        help="Path to save the boxplot image"
    )
    args = parser.parse_args()

    plot_overhead(args.csv_path, args.output)


if __name__ == '__main__':
    main()
