import os
import numpy as np
from scipy.sparse import csr_matrix


def load_mesh_npz(path: str):
    """
    Load a processed mesh from .npz.

    Returns:
        verts: (N,3) float32 array
        faces: (M,3) int64 array
    """
    data = np.load(path)
    verts = data['vertices']
    faces = data['faces']
    return verts, faces


def save_mesh_npz(path: str, verts: np.ndarray, faces: np.ndarray):
    """
    Save mesh to .npz for later loading.
    """
    np.savez_compressed(path, vertices=verts.astype(np.float32),
                        faces=faces.astype(np.int64))


def compute_residual(H: csr_matrix,
                     u: np.ndarray,
                     target: np.ndarray = None) -> np.ndarray:
    """
    Compute curvature residual = H @ u - target.

    Args:
        H: (N,N) cotangent Laplacian
        u: (N,) log‐radius vector
        target: (N,) target curvature (default zeros)

    Returns:
        residual: (N,) array
    """
    if target is None:
        target = np.zeros_like(u)
    curv = H.dot(u)
    return curv - target


def record_metrics(u: np.ndarray,
                   H: csr_matrix,
                   method: str,
                   mesh_name: str,
                   output_dir: str):
    """
    Compute per‐vertex curvature and save to disk.

    Writes:
      {output_dir}/{method}_{mesh_name}_curv.npz

    Args:
        u: (N,) final log‐radius after solve
        H: (N,N) cotangent Laplacian
        method: name string
        mesh_name: identifier string
        output_dir: where to write curvature files
    """
    os.makedirs(output_dir, exist_ok=True)
    # curvature = H @ u
    curv = H.dot(u)
    out_path = os.path.join(output_dir, f"{method}_{mesh_name}_curv.npz")
    np.savez_compressed(out_path, curvature=curv.astype(np.float64))
    # Optionally, you could record u or other stats here
    return out_path
