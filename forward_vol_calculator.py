import pandas as pd
import numpy as np

# Assuming data_loader.py is in the same directory or accessible in PYTHONPATH
from data_loader import load_and_prepare_volatility_data

def calculate_forward_volatility(spot_vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates forward volatilities from a DataFrame of spot volatilities.
    """
    df = spot_vol_df.sort_index()
    valuation_date = df.index[0] 
    ttms = (df.index - valuation_date).days / 365.25
    
    forward_vol_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    forward_vol_df.index.name = "Maturity"
    forward_vol_df.columns.name = "Delta"

    for i in range(1, len(df.index)):
        ttm1 = ttms[i-1]
        ttm2 = ttms[i]
        
        v1_sq = df.iloc[i-1] ** 2
        v2_sq = df.iloc[i] ** 2
        
        delta_ttm = ttm2 - ttm1
        
        if delta_ttm <= 0: # Also guard against non-positive delta_ttm
            forward_variance_series = pd.Series(np.nan, index=df.columns)
        else:
            numerator = (ttm2 * v2_sq) - (ttm1 * v1_sq)
            # Ensure numerator is non-negative before division
            numerator[numerator < 0] = np.nan 
            forward_variance_series = numerator / delta_ttm
            
        forward_vol_df.iloc[i] = np.sqrt(forward_variance_series)
        
    return forward_vol_df

def reconstruct_spot_volatility_from_forwards(
        reconstructed_forward_vol_df: pd.DataFrame, 
        initial_spot_vol_first_maturity: pd.Series, 
        ttm_series: pd.Series) -> pd.DataFrame:
    """
    Reconstructs spot volatilities from a DataFrame of forward volatilities.

    Args:
        reconstructed_forward_vol_df: DataFrame of forward volatilities.
                                      The first row is expected to be NaN.
        initial_spot_vol_first_maturity: Series of spot volatilities for the first maturity.
        ttm_series: Series of Time to Maturity for each date, aligned with the index.

    Returns:
        DataFrame containing reconstructed spot volatilities.
    """
    # Ensure DataFrame is sorted by index (maturity dates)
    fwd_df = reconstructed_forward_vol_df.sort_index()
    
    # Create an empty DataFrame for reconstructed spot volatilities
    recon_spot_df = pd.DataFrame(np.nan, index=fwd_df.index, columns=fwd_df.columns)
    recon_spot_df.index.name = "Maturity"
    recon_spot_df.columns.name = "Delta"

    # Set the first row of spot volatilities
    if not fwd_df.empty:
        recon_spot_df.iloc[0] = initial_spot_vol_first_maturity
    else:
        return recon_spot_df # Return empty if input is empty

    # Iterate through maturities, starting from the second one
    for i in range(1, len(fwd_df.index)):
        ttm1 = ttm_series.iloc[i-1]
        ttm2 = ttm_series.iloc[i]
        
        v_f_sq = fwd_df.iloc[i] ** 2  # Squared forward volatility for period T1 to T2
        v1_spot_sq = recon_spot_df.iloc[i-1] ** 2 # Squared spot volatility at T1
        
        delta_ttm = ttm2 - ttm1

        if ttm2 <= 0: # Denominator TTM2 must be positive
            v2_spot_sq_series = pd.Series(np.nan, index=fwd_df.columns)
        elif delta_ttm < 0: # Should not happen with sorted TTMs
             v2_spot_sq_series = pd.Series(np.nan, index=fwd_df.columns)
        else:
            # Formula: V2_spot^2 = (V_f^2 * (TTM2 - TTM1) + TTM1 * V1_spot^2) / TTM2
            # Handle cases where v_f_sq or v1_spot_sq might be NaN (e.g. if forward vol was NaN)
            term1 = v_f_sq * delta_ttm
            term2 = ttm1 * v1_spot_sq
            
            # If any input to the sum is NaN, the sum will be NaN for that element
            v2_spot_sq_numerator = term1 + term2
            
            v2_spot_sq_series = v2_spot_sq_numerator / ttm2
            # Ensure variance is not negative before taking sqrt
            v2_spot_sq_series[v2_spot_sq_series < 0] = 0 # Or np.nan if preferred for negative results
        
        recon_spot_df.iloc[i] = np.sqrt(v2_spot_sq_series)
        
    return recon_spot_df

if __name__ == "__main__":
    print("Loading initial volatility data...")
    initial_vol_df = load_and_prepare_volatility_data()
    
    print("\nOriginal Spot Volatility DataFrame (head):")
    print(initial_vol_df.head())
    
    input_df_for_forward_calc = initial_vol_df

    print(f"\nCalculating forward volatilities using: {type(input_df_for_forward_calc).__name__}")
    forward_vols_calculated = calculate_forward_volatility(input_df_for_forward_calc)
    
    print("\nCalculated Forward Volatility DataFrame (head):")
    print(forward_vols_calculated.head())

    print("\n--- Testing Spot Volatility Reconstruction ---")
    # For this test, we use the calculated forward vols as if they were "reconstructed"
    # This means if the logic is correct, we should get back the original spot vols.
    simulated_reconstructed_fwd_vols = forward_vols_calculated.copy()

    # Get the spot volatilities for the very first maturity from the original data
    initial_spot_vols_first_maturity_series = input_df_for_forward_calc.iloc[0]
    
    # Recalculate TTMs (as done in calculate_forward_volatility)
    # Ensure TTM series is aligned with the forward_vols_calculated index
    valuation_date_for_ttm = input_df_for_forward_calc.index[0]
    ttm_series_for_recon = (input_df_for_forward_calc.index - valuation_date_for_ttm).days / 365.25
    # Convert to Pandas Series with matching index
    ttm_series_for_recon = pd.Series(ttm_series_for_recon, index=input_df_for_forward_calc.index)


    print("\nInitial spot vols for first maturity (used for reconstruction):")
    print(initial_spot_vols_first_maturity_series.head())
    print("\nTTM series (head, used for reconstruction):")
    print(ttm_series_for_recon.head())

    reconstructed_spot_vols = reconstruct_spot_volatility_from_forwards(
        simulated_reconstructed_fwd_vols,
        initial_spot_vols_first_maturity_series,
        ttm_series_for_recon
    )
    
    print("\nReconstructed Spot Volatility DataFrame (head):")
    print(reconstructed_spot_vols.head())
    
    print("\nComparison: Original vs Reconstructed Spot Volatilities (norm of difference for first 5 rows):")
    # Compare relevant part (e.g. first 5 rows)
    original_head = input_df_for_forward_calc.head()
    reconstructed_head = reconstructed_spot_vols.head()
    
    # Align DataFrames just in case, though indices should match
    original_aligned, reconstructed_aligned = original_head.align(reconstructed_head, join='inner', axis=0)
    
    if not original_aligned.empty:
        diff_df_spot = original_aligned - reconstructed_aligned
        print(diff_df_spot.head())
        print("\nNorm of difference for head comparison:")
        # Calculate norm for each column or overall
        print(diff_df_spot.apply(np.linalg.norm))
        print(f"Overall norm of difference for head: {np.linalg.norm(diff_df_spot.values.flatten())}")

    else:
        print("Could not align data for spot vol difference calculation.")

    print("\nForward vol calculator script finished.")
