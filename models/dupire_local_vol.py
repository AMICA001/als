"""
dupire_local_vol.py
===================
Dupire Local Volatility Surface Construction

Author: Quant PM Team
Based on: Gatheral (2004), Henry-Labordère (2008)

Inputs:
- Arbitrage-free implied volatility surface (maturity × strike or delta grid)
- Forward price curve (or spot + dividend/yield)
- Interest rate term structure

Outputs:
- Local volatility grid ∂²C/∂K² and ∂C/∂T interpolated to a smooth σ_L(T,K)

Dependencies:
- numpy, pandas, scipy, matplotlib (optional)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# --- Constants ---
EPSILON = 1e-5

# --- Helpers ---

def compute_forward_price(spot, r, q, T):
    """Forward price using continuous compounding."""
    return spot * np.exp((r - q) * T)

def bs_vega(S, K, T, sigma, r, q):
    """Black-Scholes Vega (partial derivative of price w.r.t. vol)."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

# --- Dupire Core Function ---

def compute_local_vol_surface(iv_surface: pd.DataFrame, strikes: np.ndarray, maturities: np.ndarray,
                              smooth_iv_surface: bool = False, smoothing_sigma: float = 1.0):
    """
    Given implied vols on a strike × maturity grid, compute local vol surface σ_L(T,K).
    Assumes clean, arbitrage-free vol surface.

    Parameters:
    - iv_surface: DataFrame of implied volatilities (rows=maturities, cols=strikes)
    - strikes: 1D array of strike prices
    - maturities: 1D array of maturities
    - smooth_iv_surface: If True, apply Gaussian smoothing to the IV surface before derivative calculation.
    - smoothing_sigma: Standard deviation for Gaussian kernel if smoothing is applied.
    """
    T, K = np.meshgrid(maturities, strikes, indexing='ij')
    sigma = iv_surface.values

    if smooth_iv_surface:
        sigma = gaussian_filter(sigma, sigma=smoothing_sigma)

    # Numerical derivatives handling non-uniform spacing
    # Gradient with respect to maturities (axis 0)
    grad_T_sigma = np.gradient(sigma, axis=0)
    diff_T = np.gradient(maturities)
    # Ensure diff_T is not zero to prevent division by zero, replace with EPSILON if it is
    diff_T_expanded = np.where(np.abs(diff_T[:, np.newaxis]) < EPSILON, EPSILON, diff_T[:, np.newaxis])
    dT = grad_T_sigma / diff_T_expanded

    # Gradient with respect to strikes (axis 1)
    grad_K_sigma = np.gradient(sigma, axis=1)
    diff_K = np.gradient(strikes)
    # Ensure diff_K is not zero
    diff_K_expanded = np.where(np.abs(diff_K[np.newaxis, :]) < EPSILON, EPSILON, diff_K[np.newaxis, :])
    dK = grad_K_sigma / diff_K_expanded

    # Second derivative with respect to strikes
    grad_K_dK = np.gradient(dK, axis=1)
    # dK is already scaled by diff_K, so we divide grad_K_dK by diff_K_expanded again
    dKK = grad_K_dK / diff_K_expanded

    # Prevent division by zero in the Dupire formula denominator
    # This check is for dKK component of the denominator, K**2 can also be zero if K=0
    dKK_safe = np.where(np.abs(dKK) < EPSILON, EPSILON, dKK)

    # Dupire formula
    # Ensure K is not zero where it's used in the denominator
    K_safe = np.where(np.abs(K) < EPSILON, EPSILON, K)
    numerator = dT # dT is already sigma_t * T + sigma
    denominator = 0.5 * K_safe**2 * dKK_safe
    local_var = numerator / denominator
    local_vol = np.sqrt(np.maximum(local_var, 0))

    return pd.DataFrame(local_vol, index=iv_surface.index, columns=iv_surface.columns)
    denominator = 0.5 * K**2 * dKK
    local_var = numerator / denominator
    local_vol = np.sqrt(np.maximum(local_var, 0))

    return pd.DataFrame(local_vol, index=iv_surface.index, columns=iv_surface.columns)

# --- Visualization ---

def plot_local_vol_surface(local_vol_df):
    """3D plot of local vol surface."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    T, K = np.meshgrid(local_vol_df.columns, local_vol_df.index)
    ax.plot_surface(K, T, local_vol_df.values, cmap='viridis')
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Local Vol")
    plt.title("Dupire Local Volatility Surface")
    plt.tight_layout()
    plt.show()

# --- Example Usage ---

if __name__ == "__main__":
    # Example Usage:
    # The input iv_df should ideally be an arbitrage-free, market-observed,
    # or well-calibrated implied volatility surface.
    # Using slightly non-uniform strikes and maturities to demonstrate handling.
    strikes = np.array([80, 90, 95, 100, 105, 110, 120, 130])
    maturities = np.array([0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # Example implied volatility surface (e.g., from a model or market data)
    # This is a synthetic example for demonstration.
    iv_matrix = 0.2 + \
                0.1 * np.exp(-((strikes - 100)[None, :]**2) / (2 * 15**2)) * \
                np.exp(-maturities[:, None] / 1.0) + \
                0.05 * np.sin(2 * np.pi * maturities[:, None]) * np.cos(np.pi * (strikes - 100)[None,:] / 20) # Adding some structure

    # Add some noise to make smoothing more apparent
    np.random.seed(42) # for reproducibility
    iv_matrix += np.random.normal(0, 0.02, iv_matrix.shape)

    iv_df = pd.DataFrame(iv_matrix, index=maturities, columns=strikes)

    print("--- Computing Local Volatility (No Smoothing) ---")
    local_vol_df_no_smooth = compute_local_vol_surface(iv_df, strikes, maturities)
    print(local_vol_df_no_smooth.head())
    # plot_local_vol_surface(local_vol_df_no_smooth) # Optionally plot this

    print("\n--- Computing Local Volatility (With Gaussian Smoothing, sigma=0.5) ---")
    local_vol_df_smooth = compute_local_vol_surface(iv_df, strikes, maturities,
                                                    smooth_iv_surface=True, smoothing_sigma=0.5)
    print(local_vol_df_smooth.head())

    # Plot the smoothed surface
    # Note: plotting requires matplotlib installed and a GUI backend if running locally.
    # In some environments, plt.show() might block or not display.
    print("\nAttempting to plot the smoothed local volatility surface...")
    try:
        plot_local_vol_surface(local_vol_df_smooth)
        print("Plot displayed or saved. Check your environment if no plot appears.")
    except Exception as e:
        print(f"Plotting failed: {e}. Matplotlib might not be configured for this environment.")
