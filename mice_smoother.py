import pandas as pd
import numpy as np

# Enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
# from sklearn.ensemble import RandomForestRegressor # Example of another regressor

# Import the data loading function from the previously created script
from data_loader import load_and_prepare_volatility_data

def apply_mice_smoothing(df: pd.DataFrame, 
                         mice_iterations: int, 
                         block_mask_iterations: int, 
                         block_size: tuple, 
                         regressor) -> pd.DataFrame:
    """
    Applies MICE smoothing to a DataFrame by iteratively masking blocks and imputing.
    """
    smoothed_df = df.copy()
    rows, cols = df.shape
    block_rows, block_cols = block_size

    for iteration in range(block_mask_iterations):
        print(f"\nStarting Block Mask Iteration {iteration + 1}/{block_mask_iterations}")
        current_iteration_df_loop_internal = smoothed_df.copy() 

        for r_start in range(0, rows, block_rows):
            for c_start in range(0, cols, block_cols):
                r_end = min(r_start + block_rows, rows)
                c_end = min(c_start + block_cols, cols)
                
                # print(f"  Masking block: Rows {r_start}-{r_end-1}, Cols {c_start}-{c_end-1}") # Verbose

                # Create a temporary copy to introduce NaNs for the current block
                df_with_nans = current_iteration_df_loop_internal.copy()
                df_with_nans.iloc[r_start:r_end, c_start:c_end] = np.nan
                
                # Initialize imputer
                imputer = IterativeImputer(estimator=regressor, 
                                           max_iter=mice_iterations, 
                                           random_state=iteration * 1000 + r_start * 100 + c_start, # More unique random state
                                           imputation_order='roman',
                                           skip_complete=False, # Process all features
                                           min_value=0.0001 # Volatilities should be positive
                                          )
                
                try:
                    imputer.fit(df_with_nans)
                    imputed_values_array = imputer.transform(df_with_nans)

                    if imputed_values_array.shape[1] != df_with_nans.shape[1]:
                        # This case should ideally not be hit if IterativeImputer is robust
                        # or if df_with_nans has no all-NaN columns (which skip_complete=False should handle)
                        print(f"    WARNING: MICE imputer column count mismatch. Input: {df_with_nans.shape[1]}, Output: {imputed_values_array.shape[1]}. Block: ({r_start}-{r_end-1}, {c_start}-{c_end-1})")
                        # Fallback strategy: Fill with original values from before this block's MICE
                        # This is a simplified recovery; more sophisticated would be to align available columns.
                        # For now, we create a full-shaped array and try to place results.
                        
                        temp_reconstructed_array = df_with_nans.copy().values # Start with NaNs where they were
                        
                        # Identify columns that were actually processed (not all NaN in df_with_nans)
                        # and try to map them from imputed_values_array
                        original_cols = df_with_nans.columns
                        processed_cols_indices_input = [
                            idx for idx, col_name in enumerate(original_cols) 
                            if not df_with_nans[col_name].isnull().all()
                        ]

                        if imputed_values_array.shape[1] == len(processed_cols_indices_input):
                            for out_col_idx, original_col_idx in enumerate(processed_cols_indices_input):
                                temp_reconstructed_array[:, original_col_idx] = imputed_values_array[:, out_col_idx]
                            current_iteration_df_loop_internal.iloc[:, :] = temp_reconstructed_array
                        else:
                            print(f"    ERROR: Mismatch persists in MICE column alignment. Skipping update for this block.")
                            # If alignment fails, current_iteration_df_loop_internal for this block is not updated with imputed_values_array
                            # It will retain values from before this specific block-masking step.
                            # Effectively, this block's masking & imputation is skipped if error.

                    else: # Shape matches, direct assignment
                        current_iteration_df_loop_internal.iloc[:, :] = imputed_values_array
                
                except Exception as e:
                    print(f"    Error during MICE imputation for block ({r_start}-{r_end-1}, {c_start}-{c_end-1}): {e}")
                    # If error, this block in current_iteration_df_loop_internal remains as it was before this MICE step.
                    # (i.e., the df_with_nans for this block does not get its imputed values assigned back)

        smoothed_df = current_iteration_df_loop_internal
        print(f"Finished Block Mask Iteration {iteration + 1}/{block_mask_iterations}")

    return smoothed_df.astype(float) # Ensure float type

if __name__ == "__main__":
    print("Loading data...")
    original_df = load_and_prepare_volatility_data() 

    print("\nOriginal DataFrame (head):")
    print(original_df.head())
    
    mice_iters = 5 # Reduced for faster test
    block_mask_iters = 1 
    # Test masking the whole DataFrame to see how imputer handles potentially all-NaN columns
    block_dim = (original_df.shape[0], original_df.shape[1]) 

    reg = LinearRegression() 

    print(f"\nStarting MICE smoothing with parameters: mice_iterations={mice_iters}, "
          f"block_mask_iterations={block_mask_iters}, block_size={block_dim}, regressor={type(reg).__name__}")
          
    smoothed_df = apply_mice_smoothing(original_df, 
                                       mice_iterations=mice_iters, 
                                       block_mask_iterations=block_mask_iters, 
                                       block_size=block_dim, 
                                       regressor=reg)
    
    print("\nOriginal DataFrame (first 5x5 block):")
    print(original_df.iloc[:5, :5])
    
    print("\nSmoothed DataFrame (first 5x5 block):")
    print(smoothed_df.iloc[:5, :5])

    difference_df = smoothed_df.iloc[:5, :5] - original_df.iloc[:5, :5]
    print("\nDifference (Smoothed - Original) for the first 5x5 block:")
    print(difference_df)

    print("\nOriginal DataFrame Info:")
    original_df.info()
    print("\nSmoothed DataFrame Info:")
    smoothed_df.info()
