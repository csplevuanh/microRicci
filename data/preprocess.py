import argparse
import os
import logging

import numpy as np
import open3d as o3d

def process_mesh(input_path: str, output_dir: str, target_faces: int):
    """Load a mesh, decimate if needed, and save as .npz."""
    logging.info(f"Reading mesh: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        logging.warning(f"Failed to read mesh or mesh is empty: {input_path}")
        return

    mesh.compute_vertex_normals()

    n_tri = np.asarray(mesh.triangles).shape[0]
    if target_faces and n_tri > target_faces:
        logging.info(f"Decimating from {n_tri} → {target_faces} triangles")
        mesh = mesh.simplify_quadric_decimation(target_faces)
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}.npz")
    np.savez_compressed(out_path, vertices=verts, faces=faces)
    logging.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw mesh files into .npz (vertices, faces)."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to raw meshes directory."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Directory to write processed .npz files."
    )
    parser.add_argument(
        "--target-faces", "-t", type=int, default=10000,
        help="Max triangle count; larger meshes will be decimated to this."
    )
    parser.add_argument(
        "--extensions", "-e", nargs="+",
        default=[".obj", ".ply", ".off", ".stl"],
        help="Mesh file extensions to process."
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    os.makedirs(args.output, exist_ok=True)

    for root, _, files in os.walk(args.input):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in args.extensions):
                in_path = os.path.join(root, fname)
                try:
                    process_mesh(in_path, args.output, args.target_faces)
                except Exception as ex:
                    logging.error(f"Error processing {in_path}: {ex}")

if __name__ == "__main__":
    main()
