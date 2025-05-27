import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Import functions from other scripts
from data_loader import load_and_prepare_volatility_data
from forward_vol_calculator import calculate_forward_volatility

def apply_pca_to_forward_vol(forward_vol_df: pd.DataFrame, 
                             n_components: int) -> tuple[pd.DataFrame, PCA]:
    """
    Applies PCA to the forward volatility DataFrame.
    """
    data_for_pca = forward_vol_df.iloc[1:].copy() 
    data_for_pca.dropna(how='all', inplace=True) 
    
    if data_for_pca.empty:
        raise ValueError("DataFrame is empty after handling NaNs for PCA. Cannot apply PCA.")
    
    if data_for_pca.isnull().values.any():
        print("PCA_ANALYZER: Warning: NaNs found in data_for_pca. Filling with column means before PCA.")
        col_means = data_for_pca.mean()
        col_means.fillna(0, inplace=True) 
        data_for_pca.fillna(col_means, inplace=True)
        
        if data_for_pca.isnull().values.any():
            print("PCA_ANALYZER: Error: NaNs still present after fillna. Filling with 0.")
            data_for_pca.fillna(0, inplace=True) 

    pca_model = PCA(n_components=n_components, svd_solver='full')
    # Ensure no new NaNs before fit_transform
    if np.isnan(data_for_pca.values).any(): # Check underlying numpy array
        print("PCA_ANALYZER: ERROR - NaNs exist in data_for_pca values right before fit_transform. Filling with 0 globally.")
        data_for_pca.fillna(0, inplace=True)

    principal_components = pca_model.fit_transform(data_for_pca)
    
    pc_column_names = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    principal_components_df = pd.DataFrame(data=principal_components, 
                                           columns=pc_column_names, 
                                           index=data_for_pca.index)
    
    return principal_components_df, pca_model

def inverse_transform_pca(pca_components_df: pd.DataFrame, 
                          pca_object: PCA, 
                          original_forward_vol_columns: pd.Index,
                          original_forward_vol_index_first_row: pd.Timestamp) -> pd.DataFrame:
    """
    Reconstructs the forward volatility DataFrame from principal components.
    """
    print(f"PCA_ANALYZER: inverse_transform_pca: Input pca_components_df head:\n{pca_components_df.head()}")
    reconstructed_data_output = pca_object.inverse_transform(pca_components_df) 
    
    print(f"PCA_ANALYZER: inverse_transform_pca: reconstructed_data_output shape: {reconstructed_data_output.shape}")
    # Removed problematic .dtype print. Check if it's DataFrame or ndarray for sample printing
    if isinstance(reconstructed_data_output, pd.DataFrame):
        print(f"PCA_ANALYZER: inverse_transform_pca: reconstructed_data_output is DataFrame. Sample head:\n{reconstructed_data_output.head()}")
    elif isinstance(reconstructed_data_output, np.ndarray):
        print(f"PCA_ANALYZER: inverse_transform_pca: reconstructed_data_output is ndarray. Sample slice:\n{reconstructed_data_output[:5, :5]}") # Print a slice
    else:
        print(f"PCA_ANALYZER: inverse_transform_pca: reconstructed_data_output is of unexpected type: {type(reconstructed_data_output)}")


    print(f"PCA_ANALYZER: inverse_transform_pca: reconstructed_data_output NaNs: {np.isnan(reconstructed_data_output).sum()}, Infs: {np.isinf(reconstructed_data_output).sum()}")

    df_index = list(pca_components_df.index)
    df_columns = list(original_forward_vol_columns)

    # Critical step: creating the DataFrame
    # Ensure reconstructed_data_output is explicitly a NumPy array if it's not already, for consistency
    if isinstance(reconstructed_data_output, pd.DataFrame):
        data_for_df = reconstructed_data_output.values
    else:
        data_for_df = reconstructed_data_output

    reconstructed_df_no_first_row = pd.DataFrame(
        data=data_for_df,  # Use the (potentially converted) NumPy array
        index=df_index, 
        columns=df_columns
    )
    
    print(f"PCA_ANALYZER: inverse_transform_pca: Reconstructed DataFrame (no first NaN row) head:\n{reconstructed_df_no_first_row.head()}")
    print(f"PCA_ANALYZER: inverse_transform_pca: Reconstructed DataFrame (no first NaN row) NaNs: {reconstructed_df_no_first_row.isnull().sum().sum()}")

    first_row_df = pd.DataFrame(
        np.nan, 
        index=[original_forward_vol_index_first_row], 
        columns=df_columns 
    )
    if pca_components_df.index.name is not None: 
        first_row_df.index.name = pca_components_df.index.name

    reconstructed_forward_vol_df = pd.concat([first_row_df, reconstructed_df_no_first_row])
    print(f"PCA_ANALYZER: inverse_transform_pca: Final reconstructed_forward_vol_df head:\n{reconstructed_forward_vol_df.head()}")
    
    return reconstructed_forward_vol_df

if __name__ == '__main__':
    print("PCA_ANALYZER: Loading initial volatility data...")
    # ... (rest of main block remains the same as in Turn 37) ...
    initial_vol_df = load_and_prepare_volatility_data()
    
    print("\nPCA_ANALYZER: Calculating forward volatilities...")
    forward_vols_df = calculate_forward_volatility(initial_vol_df)
    
    original_fwd_cols = forward_vols_df.columns
    original_fwd_first_row_idx = forward_vols_df.index[0]

    print("\nPCA_ANALYZER: Original Forward Volatilities DataFrame (head):")
    print(forward_vols_df.head())
    
    data_fed_to_pca = forward_vols_df.iloc[1:].copy()
    print("\nPCA_ANALYZER: Data to be fed to PCA (forward_vols_df.iloc[1:]) (head):")
    if data_fed_to_pca.isnull().values.any(): 
        print("PCA_ANALYZER: Warning: Data fed to PCA contains NaNs BEFORE apply_pca_to_forward_vol's internal fillna.")
    print(data_fed_to_pca.head())

    num_components = 3 
    effective_samples = data_fed_to_pca.dropna(how='all').shape[0]
    effective_features = forward_vols_df.shape[1]
    
    if effective_samples == 0:
        print("PCA_ANALYZER: No data available for PCA after initial cleaning. Exiting.")
    else:
        actual_n_components = min(num_components, effective_samples, effective_features)
        if actual_n_components < num_components:
            print(f"\nPCA_ANALYZER: Warning: Requested {num_components} components, adjusted to {actual_n_components}.")
        else:
            print(f"\nPCA_ANALYZER: Applying PCA with n_components = {actual_n_components}...")

        if actual_n_components > 0:
            try:
                principal_components_df, pca_model = apply_pca_to_forward_vol(
                    forward_vols_df.copy(), 
                    n_components=actual_n_components
                )
                
                print("\nPCA_ANALYZER: Principal Components DataFrame (head):")
                print(principal_components_df.head())
                
                print("\n--- PCA_ANALYZER: Inverse PCA Transformation ---")
                reconstructed_fwd_vols_df = inverse_transform_pca(
                    principal_components_df,
                    pca_model,
                    original_fwd_cols,
                    original_fwd_first_row_idx
                )
                print("\nPCA_ANALYZER: Reconstructed Forward Volatilities DataFrame (head from main):")
                print(reconstructed_fwd_vols_df.head())
                
                original_data_for_comparison = forward_vols_df.iloc[1:]
                reconstructed_data_for_comparison = reconstructed_fwd_vols_df.iloc[1:]
                original_aligned, reconstructed_aligned = original_data_for_comparison.align(reconstructed_data_for_comparison, join='inner', axis=0)
                
                if not original_aligned.empty:
                    diff_df = original_aligned - reconstructed_aligned
                    print("\nPCA_ANALYZER: Difference (Original Fwd Vol (PCA input) - Reconstructed Fwd Vol (PCA output)) (head):")
                    print(diff_df.head())
                    print(f"\nPCA_ANALYZER: Norm of the difference: {np.linalg.norm(diff_df.fillna(0).values.flatten())}")
                else:
                    print("\nPCA_ANALYZER: Could not align data for difference calculation.")
            except ValueError as ve:
                print(f"PCA_ANALYZER: Error during PCA process: {ve}")
            except Exception as e:
                print(f"PCA_ANALYZER: An unexpected error occurred: {e}")
        else:
            print("PCA_ANALYZER: Cannot perform PCA with 0 components.")
    print("\nPCA_ANALYZER: script finished.")
