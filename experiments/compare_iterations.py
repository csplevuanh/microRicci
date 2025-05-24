import argparse
import os
import json
import pandas as pd


def collect_results(input_dir: str):
    """
    Walk through input_dir, read JSON files, and collect records.
    Returns a list of dicts with keys method, mesh, iterations, time_sec.
    """
    records = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith('.json'):
                path = os.path.join(root, fname)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    records.append({
                        'method': data.get('method', 'UNKNOWN'),
                        'mesh': data.get('mesh', os.path.splitext(fname)[0]),
                        'iterations': data.get('iterations', None),
                        'time_sec': data.get('time', None)
                    })
                except Exception as e:
                    print(f"Warning: failed to parse {path}: {e}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate iteration counts and runtimes into CSV.'
    )
    parser.add_argument(
        '--input-dir', '-i', required=True,
        help='Directory containing per-method JSON result logs.'
    )
    parser.add_argument(
        '--output-csv', '-o', default='compare_iterations.csv',
        help='Path to write aggregated CSV file.'
    )
    args = parser.parse_args()

    print(f"Reading results from {args.input_dir}...")
    records = collect_results(args.input_dir)
    if not records:
        print("No JSON logs found. Check your --input-dir.")
        return

    df = pd.DataFrame.from_records(records)
    # Ensure columns in order
    df = df[['method', 'mesh', 'iterations', 'time_sec']]
    df.to_csv(args.output_csv, index=False)
    print(f"Aggregated {len(df)} records. Saved to {args.output_csv}.")


if __name__ == '__main__':
    main()
