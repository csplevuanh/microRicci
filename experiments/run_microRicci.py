import argparse
import os
import json
import time

import numpy as np
from microRicci.utils import load_mesh_npz
from microRicci.self_tuning_flow import SelfTuningRicciFlow  # adjust if your class/name differs

def run_microRicci(verts: np.ndarray,
                   faces: np.ndarray,
                   tol: float = 1e-6,
                   max_iters: int = 1000):
    """
    Run the self-tuning greedy Ricci flow solver.
    Returns (iterations, elapsed_time_sec).
    """
    solver = SelfTuningRicciFlow(tol=tol, max_iters=max_iters)
    start = time.time()
    iterations = solver.solve(verts, faces)
    elapsed = time.time() - start
    return iterations, elapsed

def main():
    parser = argparse.ArgumentParser(
        description="Run MicroRicci self-tuning solver on processed meshes."
    )
    parser.add_argument(
        '--mesh-dir', '-m', required=True,
        help="Directory of processed .npz meshes."
    )
    parser.add_argument(
        '--output-dir', '-o', default='results/microRicci',
        help="Where to write per-run JSON logs."
    )
    parser.add_argument(
        '--tol', type=float, default=1e-6,
        help="Convergence tolerance."
    )
    parser.add_argument(
        '--max-iters', type=int, default=1000,
        help="Maximum solver iterations."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.mesh_dir)):
        if not fname.endswith('.npz'):
            continue
        mesh_name = os.path.splitext(fname)[0]
        mesh_path = os.path.join(args.mesh_dir, fname)
        verts, faces = load_mesh_npz(mesh_path)

        print(f"[microRicci] Processing {mesh_name}...")
        iters, elapsed = run_microRicci(
            verts, faces, tol=args.tol, max_iters=args.max_iters
        )

        out = {
            'method': 'microRicci',
            'mesh': mesh_name,
            'iterations': iters,
            'time': elapsed
        }
        out_path = os.path.join(
            args.output_dir,
            f"microRicci_{mesh_name}.json"
        )
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  → saved log to {out_path}")

if __name__ == '__main__':
    main()
