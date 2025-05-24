import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def build_cotangent_laplacian(verts: np.ndarray,
                              faces: np.ndarray) -> csr_matrix:
    """
    Construct the cotangent Laplacian H for a mesh.

    H[i, j] = - (cot alpha_ij + cot beta_ij) / 2  for i != j (if edge ij exists)
    H[i, i] = -sum_{j != i} H[i, j]

    Args:
        verts: (N, 3) array of vertex coordinates.
        faces: (M, 3) array of triangle indices (0-based).

    Returns:
        H: (N, N) sparse CSR matrix of cotangent Laplacian.
    """
    # For each triangle, compute cotangents at its three corners
    I = []
    J = []
    W = []

    def cotangent(a, b, c):
        # angle at vertex a of triangle (a, b, c)
        u = b - a
        v = c - a
        # cot θ = (u·v) / ||u × v||
        cross = np.cross(u, v)
        denom = np.linalg.norm(cross)
        if denom < 1e-12:
            return 0.0
        return np.dot(u, v) / denom

    for tri in faces:
        i, j, k = tri
        vi, vj, vk = verts[i], verts[j], verts[k]

        cot_alpha = cotangent(vj, vk, vi)  # at vi opposite edge jk
        cot_beta  = cotangent(vk, vi, vj)  # at vj opposite edge ki
        cot_gamma = cotangent(vi, vj, vk)  # at vk opposite edge ij

        # accumulate weights for edges (j,k), (k,i), (i,j)
        for (p, q, w) in [(j, k, cot_alpha),
                          (k, i, cot_beta),
                          (i, j, cot_gamma)]:
            # symmetric entries
            I.extend([p, q])
            J.extend([q, p])
            W.extend([w * 0.5, w * 0.5])

    # Assemble off-diagonal entries
    N = verts.shape[0]
    L = coo_matrix((W, (I, J)), shape=(N, N))

    # Compute diagonal: negative row-sum of off-diagonals
    diag_weights = -np.array(L.sum(axis=1)).ravel()
    diag_I = np.arange(N)
    diag_J = np.arange(N)
    diag_W = diag_weights

    # Final Laplacian = off-diagonal + diag
    H = coo_matrix((diag_W, (diag_I, diag_J)), shape=(N, N)) + L
    return H.tocsr()
