import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .laplacian import build_cotangent_laplacian


class GreedySolver:
    def __init__(self, tol: float = 1e-6, max_iters: int = 1000):
        """
        Args:
            tol: convergence tolerance on max |residual|.
            max_iters: maximum number of iterations before stopping.
        """
        self.tol = tol
        self.max_iters = max_iters

    def solve(self, verts: np.ndarray, faces: np.ndarray):
        """
        Run greedy Ricci flow until convergence.
        
        Args:
            verts: (N,3) array of vertex positions.
            faces: (M,3) array of triangle indices.
        
        Returns:
            iters: number of iterations performed.
        """
        # Build cotangent Laplacian H (sparse NxN) and mass matrix if needed
        H = build_cotangent_laplacian(verts, faces)  # returns scipy.sparse matrix
        N = verts.shape[0]

        # Compute target curvatures: here zero curvature everywhere for uniformization
        target = np.zeros(N, dtype=np.float64)

        # Initial curvature: H @ u where u is log radius (start at zeros)
        u = np.zeros(N, dtype=np.float64)
        curv = H.dot(u)

        # Residual = curv - target
        res = curv - target

        for it in range(1, self.max_iters + 1):
            # Check convergence
            max_res = np.max(np.abs(res))
            if max_res < self.tol:
                return it

            # Pick vertex with largest residual magnitude
            idx = np.argmax(np.abs(res))

            # Solve for local update: H_row * delta_u = -res[idx]
            # Extract row idx of H
            H_row = H.getrow(idx).toarray().ravel()
            diag = H_row[idx]
            if np.abs(diag) < 1e-12:
                # Ill-conditioned, break out
                return it

            # Simple step: delta = -residual / diag
            delta = -res[idx] / diag

            # Update u at idx
            u[idx] += delta

            # Update curvature and residual incrementally:
            # curv_new = curv_old + H[:,idx] * delta
            col = H.getcol(idx).toarray().ravel()
            curv += col * delta
            res = curv - target

        return self.max_iters
