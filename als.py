import numpy as np

def ALS_MatrixCompletion(data, rank=2, max_iter=50, tol=0.001):
    """
    Alternating Least Squares (ALS) Matrix Completion.

    Parameters:
    - data: 2D numpy array or list of lists with missing values as np.nan
    - rank: int, number of latent features
    - max_iter: int, maximum number of iterations
    - tol: float, tolerance for stopping criterion

    Returns:
    - result: 2D numpy array, completed matrix
    """
    data = np.array(data, dtype=float)
    m, n = data.shape

    # Mask for observed (non-nan) entries
    mask = ~np.isnan(data)
    filled = np.where(mask, data, 0.0)

    # Initialize U and V randomly
    rng = np.random.default_rng()
    U = rng.random((m, rank))
    V = rng.random((n, rank))

    for iteration in range(max_iter):
        change = 0.0

        # Update U
        for i in range(m):
            for k in range(rank):
                numeratorU = 0.0
                denominatorU = 0.0
                for j in range(n):
                    if mask[i, j]:
                        pred = sum(U[i, r] * V[j, r] for r in range(rank) if r != k)
                        numeratorU += V[j, k] * (filled[i, j] - pred)
                        denominatorU += V[j, k] ** 2
                if denominatorU > 0:
                    diff = U[i, k]
                    U[i, k] = numeratorU / denominatorU
                    change += abs(U[i, k] - diff)

        # Update V
        for j in range(n):
            for k in range(rank):
                numeratorV = 0.0
                denominatorV = 0.0
                for i in range(m):
                    if mask[i, j]:
                        pred = sum(U[i, r] * V[j, r] for r in range(rank) if r != k)
                        numeratorV += U[i, k] * (filled[i, j] - pred)
                        denominatorV += U[i, k] ** 2
                if denominatorV > 0:
                    diff = V[j, k]
                    V[j, k] = numeratorV / denominatorV
                    change += abs(V[j, k] - diff)

        if change < tol:
            break

    # Reconstruct the completed matrix
    result = np.dot(U, V.T)
    return result
