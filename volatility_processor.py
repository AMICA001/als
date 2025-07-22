import pandas as pd
import numpy as np
import torch

# Import functions from other modules
from data_loader import load_and_prepare_volatility_data
from mice_smoother import apply_mice_smoothing
from forward_vol_calculator import calculate_forward_volatility, reconstruct_spot_volatility_from_forwards
from pca_analyzer import apply_pca_to_forward_vol, inverse_transform_pca
from surface_fitter import fit_quadratic_surface
from bigan import train_bigan

# Regressor for MICE
from sklearn.linear_model import LinearRegression # Using LinearRegression for speed; BayesianRidge is also good.

def run_volatility_processing_pipeline(
    initial_data_str: str,
    num_pipeline_iterations: int,
    mice_iterations: int,
    block_mask_iterations: int,
    block_size: tuple,
    mice_regressor, # e.g., LinearRegression()
    pca_n_components: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the iterative volatility processing workflow.
    """
    print("--- Starting Volatility Processing Pipeline ---")

    # 1. Initial Data Load
    current_spot_vol_df = load_and_prepare_volatility_data(data_str=initial_data_str)
    original_spot_vol_df = current_spot_vol_df.copy()
    print("Initial data loaded and prepared.")
    print(f"Shape of initial spot volatility data: {current_spot_vol_df.shape}")

    base_df_for_ttm = original_spot_vol_df.sort_index()
    valuation_date_for_ttm = base_df_for_ttm.index[0]
    ttm_series = (base_df_for_ttm.index - valuation_date_for_ttm).days / 365.25
    ttm_series = pd.Series(ttm_series, index=base_df_for_ttm.index)
    ttm_series.name = "TTM"
    # print("TTM Series calculated (head):") # Reduced verbosity
    # print(ttm_series.head())

    for i_pipeline in range(num_pipeline_iterations):
        print(f"\n--- Pipeline Iteration {i_pipeline + 1}/{num_pipeline_iterations} ---")
        initial_spot_vol_first_maturity = current_spot_vol_df.iloc[0].copy()
        # print(f"Iteration {i_pipeline+1}: Stored spot vols for first maturity.") # Reduced verbosity

        # print(f"Iteration {i_pipeline+1}: Applying MICE smoothing...") # Reduced verbosity
        current_spot_vol_df_sorted = current_spot_vol_df.sort_index()
        smoothed_spot_df = apply_mice_smoothing(
            current_spot_vol_df_sorted,
            mice_iterations=mice_iterations,
            block_mask_iterations=block_mask_iterations,
            block_size=block_size,
            regressor=mice_regressor
        )
        current_spot_vol_df = smoothed_spot_df.reindex(columns=original_spot_vol_df.columns, index=original_spot_vol_df.index)
        # print(f"Iteration {i_pipeline+1}: MICE smoothing complete.") # Reduced verbosity

        forward_vol_df = calculate_forward_volatility(current_spot_vol_df)
        # print(f"Iteration {i_pipeline+1}: Forward volatility calculation complete.") # Reduced verbosity

        original_fwd_cols = forward_vol_df.columns
        original_fwd_first_row_idx = forward_vol_df.index[0] 

        pca_input_data = forward_vol_df.iloc[1:].dropna(how='all')
        effective_samples = pca_input_data.shape[0]
        effective_features = pca_input_data.shape[1]
        
        if effective_samples == 0 or effective_features == 0:
            print(f"Iteration {i_pipeline+1}: Skipping PCA and reconstruction due to empty data for PCA.")
            continue 

        actual_pca_n_components = min(pca_n_components, effective_samples, effective_features)
        if actual_pca_n_components < pca_n_components:
             print(f"Iteration {i_pipeline+1}: PCA n_components adjusted to {actual_pca_n_components}.")
        if actual_pca_n_components == 0:
            print(f"Iteration {i_pipeline+1}: Skipping PCA and reconstruction (0 components).")
            continue

        # print(f"Iteration {i_pipeline+1}: Applying PCA with {actual_pca_n_components} components...") # Reduced verbosity
        pca_components_df, pca_object = apply_pca_to_forward_vol(
            forward_vol_df.copy(), 
            n_components=actual_pca_n_components
        )
        # print(f"Iteration {i_pipeline+1}: PCA complete.") # Reduced verbosity

        reconstructed_forward_vol_df = inverse_transform_pca(
            pca_components_df,
            pca_object,
            original_fwd_cols,
            original_fwd_first_row_idx
        )
        # print(f"Iteration {i_pipeline+1}: Inverse PCA complete.") # Reduced verbosity
        
        current_spot_vol_df = reconstruct_spot_volatility_from_forwards(
            reconstructed_forward_vol_df,
            initial_spot_vol_first_maturity, 
            ttm_series 
        )
        print(f"Iteration {i_pipeline+1}: Spot volatility reconstruction complete. Processed head:\n{current_spot_vol_df.head()}")

    print("\n--- Volatility Processing Pipeline Finished ---")
    return current_spot_vol_df, original_spot_vol_df

def run_bigan_pipeline(
    vol_surface_df: pd.DataFrame,
    market_conditions_df: pd.DataFrame,
    latent_dim: int = 4,
    n_epochs: int = 200,
    lr: float = 0.0002,
    b1: float = 0.5,
    b2: float = 0.999
):
    """
    Orchestrates the BiGAN-based volatility surface modeling pipeline.
    """
    print("--- Starting BiGAN Pipeline ---")

    # 1. Fit quadratic surface and get residuals
    param_df, residual_df, modeled_df = fit_quadratic_surface(vol_surface_df)
    print("Quadratic surface fitting complete.")

    # 2. Train BiGAN on residuals
    print("Training BiGAN on residual surfaces...")
    generator, encoder = train_bigan(
        residual_df,
        market_conditions_df,
        latent_dim=latent_dim,
        n_epochs=n_epochs,
        lr=lr,
        b1=b1,
        b2=b2
    )
    print("BiGAN training complete.")

    # 3. Generate new surfaces (example)
    # For demonstration, we'll generate surfaces based on the training conditions
    z = torch.randn(market_conditions_df.shape[0], latent_dim)
    generated_residuals = generator(z, torch.from_numpy(market_conditions_df.values).float())

    # Reshape to match the original format
    generated_residuals_df = pd.DataFrame(
        generated_residuals.detach().numpy().reshape(residual_df.shape),
        index=residual_df.index,
        columns=residual_df.columns
    )

    # 4. Reconstruct full surfaces
    reconstructed_surfaces_df = modeled_df + generated_residuals_df
    print("--- BiGAN Pipeline Finished ---")

    return reconstructed_surfaces_df, param_df, residual_df, modeled_df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Volatility Surface Modeling Pipeline")
    parser.add_argument('pipeline', choices=['pca', 'bigan'], help="Which pipeline to run.")
    parser.add_argument('--vol_surface_file', type=str, default='vol_surface.csv', help="Path to the volatility surface CSV file.")
    parser.add_argument('--market_conditions_file', type=str, default='market_conditions.csv', help="Path to the market conditions CSV file.")
    parser.add_argument('--param_fit_file', type=str, default='param_fit.csv', help="Path to the parametric fit CSV file (optional).")
    parser.add_argument('--latent_dim', type=int, default=4, help="Latent dimension for the BiGAN model.")
    parser.add_argument('--n_epochs', type=int, default=200, help="Number of epochs for BiGAN training.")
    parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate for BiGAN training.")
    parser.add_argument('--b1', type=float, default=0.5, help="Adam optimizer beta 1 for BiGAN training.")
    parser.add_argument('--b2', type=float, default=0.999, help="Adam optimizer beta 2 for BiGAN training.")
    parser.add_argument('--num_pipeline_iterations', type=int, default=2, help="Number of iterations for the PCA pipeline.")
    parser.add_argument('--mice_iterations', type=int, default=10, help="Number of MICE iterations for the PCA pipeline.")
    parser.add_argument('--block_mask_iterations', type=int, default=1, help="Number of block mask iterations for the PCA pipeline.")
    parser.add_argument('--pca_n_components', type=int, default=3, help="Number of PCA components for the PCA pipeline.")

    args = parser.parse_args()

    if args.pipeline == 'pca':
        # Load data from the default string for now, as the CLI is for the BiGAN pipeline
        RAW_VOLATILITY_DATA = """maturity/delta vol	1	10	20	30	40	50	60	70	80	90	99
25-Jul-25	29.67%	24.10%	21.63%	19.72%	17.96%	16.32%	14.83%	13.63%	12.84%	12.57%	15.71%
31-Jul-25	30.02%	24.34%	21.85%	19.92%	18.14%	16.45%	14.92%	13.68%	12.87%	12.60%	15.81%
15-Aug-25	30.68%	24.76%	22.23%	20.27%	18.44%	16.68%	15.05%	13.71%	12.81%	12.51%	15.67%
29-Aug-25	31.10%	25.09%	22.54%	20.55%	18.70%	16.90%	15.20%	13.75%	12.74%	12.39%	15.70%
19-Sep-25	31.72%	25.53%	22.95%	20.94%	19.05%	17.19%	15.40%	13.85%	12.72%	12.38%	15.76%
30-Sep-25	31.88%	25.64%	23.05%	21.04%	19.16%	17.29%	15.46%	13.84%	12.67%	12.29%	15.78%
17-Oct-25	32.20%	25.82%	23.21%	21.21%	19.33%	17.44%	15.55%	13.85%	12.62%	12.25%	15.67%
31-Oct-25	32.41%	25.98%	23.34%	21.34%	19.46%	17.57%	15.64%	13.87%	12.59%	12.24%	15.74%
21-Nov-25	32.65%	26.13%	23.47%	21.45%	19.57%	17.68%	15.71%	13.88%	12.53%	12.10%	15.72%
19-Dec-25	32.87%	26.28%	23.59%	21.56%	19.67%	17.78%	15.80%	13.93%	12.54%	12.06%	15.58%
"""
        params = {
            "initial_data_str": RAW_VOLATILITY_DATA,
            "num_pipeline_iterations": args.num_pipeline_iterations,
            "mice_iterations": args.mice_iterations,
            "block_mask_iterations": args.block_mask_iterations,
            "block_size": (2, 2),
            "mice_regressor": LinearRegression(),
            "pca_n_components": args.pca_n_components
        }
        final_processed_df, original_df = run_volatility_processing_pipeline(**params)
        original_df.to_csv("original_volatility_data.csv")
        final_processed_df.to_csv("final_processed_volatility_data.csv")
        print("PCA pipeline finished. Results saved to CSV.")

    elif args.pipeline == 'bigan':
        vol_surface_df = pd.read_csv(args.vol_surface_file, index_col='surf_id')
        market_conditions_df = pd.read_csv(args.market_conditions_file, index_col='surf_id')
        
        reconstructed_surfaces, _, _, _ = run_bigan_pipeline(
            vol_surface_df,
            market_conditions_df,
            latent_dim=args.latent_dim,
            n_epochs=args.n_epochs,
            lr=args.lr,
            b1=args.b1,
            b2=args.b2
        )
        reconstructed_surfaces.to_csv("reconstructed_volatility_surfaces.csv")
        print("BiGAN pipeline finished. Reconstructed surfaces saved to CSV.")
