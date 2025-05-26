import numpy as np
import cupy as cp

def ALS_MatrixCompletion(data, rank=2, max_iter=50, tol=0.001, regularization_term=0.01):
    """
    Alternating Least Squares (ALS) Matrix Completion using CuPy for GPU acceleration.

    Parameters:
    - data: 2D numpy array or list of lists with missing values as np.nan
    - rank: int, number of latent features
    - max_iter: int, maximum number of iterations
    - tol: float, tolerance for stopping criterion
    - regularization_term: float, regularization term for matrix factorization

    Returns:
    - result: 2D numpy array, completed matrix
    """
    data_cp = cp.asarray(data, dtype=float)
    m, n = data_cp.shape

    # Mask for observed (non-nan) entries
    mask = ~cp.isnan(data_cp)
    filled = cp.where(mask, data_cp, 0.0)

    # Initialize U and V randomly on GPU
    U = cp.random.random((m, rank))
    V = cp.random.random((n, rank))

    for iteration in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()

        # Update U
        for i in range(m):
            observed_cols = cp.where(mask[i, :])[0]
            if observed_cols.size == 0:
                continue  # Or handle as appropriate, e.g., U[i, :] = cp.random.random(rank) or cp.zeros(rank)
            
            V_obs = V[observed_cols, :]
            filled_obs_row = filled[i, observed_cols]
            
            A = V_obs.T @ V_obs + regularization_term * cp.eye(rank)
            b = V_obs.T @ filled_obs_row
            
            try:
                U[i, :] = cp.linalg.solve(A, b)
            except cp.cuda.cusolver.CUSOLVERError as e:
                # Handle potential singularity or other solver issues, e.g. by re-initializing U[i,:]
                # For now, just print a warning and keep the old value or set to zero/random
                print(f"Warning: Solver error for U[{i}, :]: {e}. Keeping old value or re-initializing.")
                # U[i, :] = U_old[i, :] # Keep old value
                # U[i, :] = cp.random.random(rank) # Re-initialize randomly
                # Or, if you want to be robust against singular A, use cp.linalg.lstsq
                # U[i, :] = cp.linalg.lstsq(A, b)[0]


        # Update V
        for j in range(n):
            observed_rows = cp.where(mask[:, j])[0]
            if observed_rows.size == 0:
                continue # Or handle as appropriate

            U_obs = U[observed_rows, :]
            filled_obs_col = filled[observed_rows, j]

            A = U_obs.T @ U_obs + regularization_term * cp.eye(rank)
            b = U_obs.T @ filled_obs_col
            
            try:
                V[j, :] = cp.linalg.solve(A, b)
            except cp.cuda.cusolver.CUSOLVERError as e:
                print(f"Warning: Solver error for V[{j}, :]: {e}. Keeping old value or re-initializing.")
                # V[j, :] = V_old[j, :]
                # V[j, :] = cp.random.random(rank)
                # V[j, :] = cp.linalg.lstsq(A, b)[0]


        change = cp.sum(cp.abs(U - U_old)) + cp.sum(cp.abs(V - V_old))

        if change < tol:
            break

    # Reconstruct the completed matrix
    result = cp.dot(U, V.T)
    
    # Ensure the function returns a NumPy array
    return cp.asnumpy(result)
