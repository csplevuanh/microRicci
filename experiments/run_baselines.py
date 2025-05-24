import argparse
import os
import json
import time

import numpy as np
from microRicci.laplacian import build_cotangent_laplacian
from microRicci.utils import load_mesh_npz

# Placeholder imports for baseline implementations:
# from baselines.orc_pool import orc_pool_solver
# from baselines.gnr_flow import gnr_flow_solver
# from baselines.learning_ricci import learning_ricci_solver

def run_orc_pool(verts, faces, tol=1e-6):
    """
    ORC-Pool solver.
    TODO: import and call your ORC-Pool implementation here.
    Should return (iterations, elapsed_time_sec).
    """
    start = time.time()
    # e.g., iterations = orc_pool_solver(verts, faces, tol=tol)
    iterations = None  # TODO
    elapsed = time.time() - start
    return iterations, elapsed

def run_gnr_flow(verts, faces, tol=1e-6):
    """
    GNRF solver.
    TODO: import and call your GNRF implementation here.
    """
    start = time.time()
    # iterations = gnr_flow_solver(verts, faces, tol=tol)
    iterations = None  # TODO
    elapsed = time.time() - start
    return iterations, elapsed

def run_learning_ricci(verts, faces, tol=1e-6):
    """
    Learning-Ricci solver.
    TODO: import and call your Learning-Ricci implementation here.
    """
    start = time.time()
    # iterations = learning_ricci_solver(verts, faces, tol=tol)
    iterations = None  # TODO
    elapsed = time.time() - start
    return iterations, elapsed

METHOD_FUNCS = {
    'orc_pool': run_orc_pool,
    'gnr_flow': run_gnr_flow,
    'learning_ricci': run_learning_ricci,
}

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline Ricci‐flow methods on processed meshes."
    )
    parser.add_argument(
        '--mesh-dir', '-m', required=True,
        help="Directory of processed .npz meshes."
    )
    parser.add_argument(
        '--output-dir', '-o', default='results/baselines',
        help="Where to write per‐run JSON logs."
    )
    parser.add_argument(
        '--methods', '-x', nargs='+',
        choices=METHOD_FUNCS.keys(),
        default=list(METHOD_FUNCS.keys()),
        help="Which baseline methods to run."
    )
    parser.add_argument(
        '--tol', type=float, default=1e-6,
        help="Convergence tolerance for all solvers."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over all .npz meshes
    for fname in sorted(os.listdir(args.mesh_dir)):
        if not fname.endswith('.npz'):
            continue
        mesh_name = os.path.splitext(fname)[0]
        mesh_path = os.path.join(args.mesh_dir, fname)
        verts, faces = load_mesh_npz(mesh_path)

        for method in args.methods:
            func = METHOD_FUNCS[method]
            print(f"[{method}] Processing {mesh_name}...")
            iterations, elapsed = func(verts, faces, tol=args.tol)

            out = {
                'method': method,
                'mesh': mesh_name,
                'iterations': iterations,
                'time': elapsed
            }
            out_path = os.path.join(
                args.output_dir,
                f"{method}_{mesh_name}.json"
            )
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  → saved log to {out_path}")

if __name__ == '__main__':
    main()
