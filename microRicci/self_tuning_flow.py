import numpy as np
from scipy.sparse import csr_matrix

from .laplacian import build_cotangent_laplacian
from .greedy_solver import GreedySolver
from .selector import SelectorMLP
from .regressor import RegressorMLP


class SelfTuningRicciFlow:
    def __init__(self,
                 tol: float = 1e-6,
                 max_iters: int = 1000,
                 selector_path: str = None,
                 regressor_path: str = None,
                 device: str = 'cpu'):
        """
        Args:
            tol: convergence tolerance on max |residual|.
            max_iters: maximum number of iterations.
            selector_path: path to trained SelectorMLP (.pt/.pth).
            regressor_path: path to trained RegressorMLP (.pt/.pth).
            device: torch device for model inference.
        """
        self.tol = tol
        self.max_iters = max_iters
        self.device = device

        # Load models if provided
        if selector_path:
            self.selector = SelectorMLP.load(selector_path, device=device)
        else:
            self.selector = None

        if regressor_path:
            self.regressor = RegressorMLP.load(regressor_path, device=device)
        else:
            self.regressor = None

        # Fallback to pure greedy if no models
        self.greedy = GreedySolver(tol=tol, max_iters=max_iters)

    def solve(self,
              verts: np.ndarray,
              faces: np.ndarray) -> int:
        """
        Run self‐tuning Ricci flow. If models are available, uses
        learned selector+regressor; otherwise falls back to greedy.
        
        Args:
            verts: (N,3) vertex positions.
            faces: (M,3) triangle indices.
        
        Returns:
            iters: number of iterations performed.
        """
        # If no selector or regressor, delegate to greedy solver
        if self.selector is None or self.regressor is None:
            return self.greedy.solve(verts, faces)

        # Build Laplacian
        H: csr_matrix = build_cotangent_laplacian(verts, faces)
        N = verts.shape[0]

        # Target curvature = 0 everywhere
        target = np.zeros(N, dtype=np.float64)

        # Log‐radius / potential
        u = np.zeros(N, dtype=np.float64)
        # Initial curvature and residual
        curv = H.dot(u)
        res = curv - target

        # Pre‐extract diagonal of H for features
        H_diag = H.diagonal()

        for it in range(1, self.max_iters + 1):
            max_res = np.max(np.abs(res))
            if max_res < self.tol:
                return it

            # Build per‐vertex features: [|residual|, diagonal_entry]
            feats = np.stack([np.abs(res), H_diag], axis=1)

            # Select vertex to update
            idx = self.selector.select_vertex(feats, device=self.device)

            # Predict step size for all vertices, then pick for idx
            # (could batch predict only one, but simpler to run full)
            delta_all = self.regressor(
                __import__('torch').from_numpy(feats.astype(np.float32))
                                  .to(self.device)
            ).cpu().numpy()
            delta = float(delta_all[idx])

            # Safety check: fallback if regressor fails
            if not np.isfinite(delta):
                # degenerate; use greedy step
                delta = -res[idx] / (H_diag[idx] + 1e-12)

            # Apply update at idx
            u[idx] += delta

            # Incrementally update curvature and residual:
            # curv_new = curv_old + H[:, idx] * delta
            col = H.getcol(idx).toarray().ravel()
            curv += col * delta
            res = curv - target

        return self.max_iters
