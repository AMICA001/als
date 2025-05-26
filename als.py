import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def _als_imputation(data, rank=2, max_iter=50, tol=0.001):
    """
    Helper function for Alternating Least Squares (ALS) Matrix Completion.
    """
    data_als = np.array(data, dtype=float)
    m, n = data_als.shape

    # Mask for observed (non-nan) entries
    mask = ~np.isnan(data_als)
    filled = np.where(mask, data_als, 0.0)

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
                for j_col in range(n):
                    if mask[i, j_col]:
                        pred = sum(U[i, r] * V[j_col, r] for r in range(rank) if r != k)
                        numeratorU += V[j_col, k] * (filled[i, j_col] - pred)
                        denominatorU += V[j_col, k] ** 2
                if denominatorU > 0:
                    diff = U[i, k]
                    U[i, k] = numeratorU / denominatorU
                    change += abs(U[i, k] - diff)

        # Update V
        for j_col in range(n):
            for k in range(rank):
                numeratorV = 0.0
                denominatorV = 0.0
                for i_row in range(m):
                    if mask[i_row, j_col]:
                        pred = sum(U[i_row, r] * V[j_col, r] for r in range(rank) if r != k)
                        numeratorV += U[i_row, k] * (filled[i_row, j_col] - pred)
                        denominatorV += U[i_row, k] ** 2
                if denominatorV > 0:
                    diff = V[j_col, k]
                    V[j_col, k] = numeratorV / denominatorV
                    change += abs(V[j_col, k] - diff)

        if change < tol:
            break

    # Reconstruct the completed matrix
    result = np.dot(U, V.T)
    # Fill original NaNs with imputed values, keep original non-NaNs
    imputed_data = np.array(data, dtype=float)
    nan_mask = np.isnan(imputed_data)
    imputed_data[nan_mask] = result[nan_mask]
    return imputed_data

def impute_missing_values(data, method='ALS', rank=2, max_iter=50, tol=0.001, estimator=None):
    """
    Imputes missing values in a dataset using either Alternating Least Squares (ALS)
    or Multivariate Imputation by Chained Equations (MICE).

    Parameters:
    - data: 2D numpy array or list of lists with missing values as np.nan.
            This is the dataset to be imputed.
    - method: str, default 'ALS'. Specifies the imputation method.
              'ALS': Uses Alternating Least Squares for matrix completion.
              'MICE': Uses Multivariate Imputation by Chained Equations.
    
    For 'ALS' method:
    - rank: int, default 2. Number of latent features for ALS.
    - max_iter: int, default 50. Maximum number of iterations for ALS.
    - tol: float, default 0.001. Tolerance for stopping criterion in ALS.

    For 'MICE' method:
    - estimator: scikit-learn regressor instance, default None.
                 The estimator to use for MICE imputation. If None, BayesianRidge() is used.
                 Users can pass other scikit-learn regressor instances. For example:
                 - BayesianRidge(): Good for small datasets or when a degree of regularization is desired.
                                    It's a probabilistic model that can handle uncertainty.
                 - DecisionTreeRegressor(): Can capture non-linear relationships in the data.
                                            However, a single tree can be prone to overfitting.
                 - RandomForestRegressor(): An ensemble of decision trees. More robust and accurate
                                            than a single decision tree, less prone to overfitting,
                                            but more computationally intensive.

    Returns:
    - result: 2D numpy array. The dataset with missing values imputed.
    """
    data_np = np.array(data, dtype=float)

    if method == 'ALS':
        return _als_imputation(data_np, rank=rank, max_iter=max_iter, tol=tol)
    elif method == 'MICE':
        if estimator is None:
            estimator = BayesianRidge()
        
        imputer = IterativeImputer(estimator=estimator, random_state=0, max_iter=max_iter, tol=tol)
        # IterativeImputer expects features as columns, so if we have a single feature (e.g. a single column vector)
        # it needs to be reshaped. However, typical use cases will have multiple features (columns).
        if data_np.ndim == 1:
             data_np_reshaped = data_np.reshape(-1, 1)
             imputed_data = imputer.fit_transform(data_np_reshaped)
             return imputed_data.flatten() # Return to original shape if it was a single column
        else:
             return imputer.fit_transform(data_np)
    else:
        raise ValueError("Invalid method. Choose 'ALS' or 'MICE'.")
