import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def fit_quadratic_surface(vol_surface_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fits a quadratic surface to each volatility surface in the input DataFrame.

    Args:
        vol_surface_df: A DataFrame where each row is a flattened volatility surface.

    Returns:
        A tuple containing:
        - A DataFrame of the fitted parameters (ATM, Skew, Slope) for each surface.
        - A DataFrame of the residual surfaces.
        - A DataFrame of the modeled surfaces.
    """
    # Create the coordinate grid
    maturities = np.array([float(col.split('_')[1]) for col in vol_surface_df.columns])
    strikes = np.array([float(col.split('_')[2]) for col in vol_surface_df.columns])

    X = np.vstack([np.ones(len(maturities)), strikes, maturities]).T

    all_params = []
    all_residuals = []
    all_modeled = []

    for index, row in vol_surface_df.iterrows():
        y = row.values

        # Fit the linear model
        model = LinearRegression()
        model.fit(X, y)

        # Get the parameters
        atm, skew, slope = model.intercept_, model.coef_[1], model.coef_[2]
        all_params.append({'surf_id': index, 'ATM': atm, 'Skew': skew, 'Slope': slope})

        # Calculate the modeled surface
        modeled_surface = model.predict(X)
        all_modeled.append(modeled_surface)

        # Calculate the residual surface
        residual_surface = y - modeled_surface
        all_residuals.append(residual_surface)

    param_df = pd.DataFrame(all_params).set_index('surf_id')
    residual_df = pd.DataFrame(all_residuals, index=vol_surface_df.index, columns=vol_surface_df.columns)
    modeled_df = pd.DataFrame(all_modeled, index=vol_surface_df.index, columns=vol_surface_df.columns)

    return param_df, residual_df, modeled_df

if __name__ == '__main__':
    # Create a dummy volatility surface for testing
    data = {
        'vol_0_0': [0.2, 0.21],
        'vol_0_1': [0.22, 0.23],
        'vol_1_0': [0.18, 0.19],
        'vol_1_1': [0.2, 0.21]
    }
    dummy_surface_df = pd.DataFrame(data)

    params, residuals, modeled = fit_quadratic_surface(dummy_surface_df)

    print("--- Fitted Parameters ---")
    print(params)
    print("\n--- Residual Surfaces ---")
    print(residuals)
    print("\n--- Modeled Surfaces ---")
    print(modeled)
