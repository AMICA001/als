import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For date formatting
import seaborn as sns

# Import functions from other scripts
from data_loader import load_and_prepare_volatility_data
from forward_vol_calculator import calculate_forward_volatility
from pca_analyzer import apply_pca_to_forward_vol # Added for PCA plotting

def plot_forward_volatility(forward_vol_df: pd.DataFrame, output_filename: str):
    """
    Plots the forward volatility curves for each delta and saves the plot.

    Args:
        forward_vol_df: DataFrame containing forward volatilities with maturities
                        as index and deltas as columns. The first row might be NaNs.
        output_filename: The path to save the plot image.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    plot_data = forward_vol_df.iloc[1:] # Skip first row if it's all NaN

    for delta_col in plot_data.columns:
        ax.plot(plot_data.index, plot_data[delta_col], marker='o', linestyle='-', label=f'Delta {delta_col}')
        
    ax.set_title('Forward Volatility by Delta', fontsize=16)
    ax.set_xlabel('Maturity Date', fontsize=12)
    ax.set_ylabel('Forward Volatility', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.legend(title='Delta Levels', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    try:
        plt.show()
    except Exception as e:
        print(f"Error showing plot (this might be normal in a non-GUI environment): {e}")

def plot_pca_components(pca_components_df: pd.DataFrame, output_filename: str):
    """
    Plots the principal components over time and saves the plot.

    Args:
        pca_components_df: DataFrame with maturities as index and 
                           principal components (PC1, PC2, etc.) as columns.
        output_filename: The path to save the plot image.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    for pc_col in pca_components_df.columns:
        ax.plot(pca_components_df.index, pca_components_df[pc_col], marker='o', linestyle='-', label=pc_col)
        
    ax.set_title('Principal Components Over Time', fontsize=16)
    ax.set_xlabel('Maturity Date', fontsize=12)
    ax.set_ylabel('Principal Component Value', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.legend(title='Principal Components')
    plt.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"PCA Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving PCA plot: {e}")

    try:
        plt.show()
    except Exception as e:
        print(f"Error showing PCA plot (this might be normal in a non-GUI environment): {e}")

if __name__ == '__main__':
    print("--- Loading initial volatility data ---")
    initial_vol_df = load_and_prepare_volatility_data()
    
    print("\n--- Calculating forward volatilities ---")
    forward_vols_df = calculate_forward_volatility(initial_vol_df)
    
    print("\n--- Plotting Forward Volatilities ---")
    print("Forward Volatilities DataFrame (head for plotting):")
    if not forward_vols_df.iloc[1:].empty:
        print(forward_vols_df.iloc[1:].head())
    else:
        print("Forward volatility data (after skipping first row) is empty for plotting.")

    output_fwd_plot_filename = "forward_volatility_curves.png"
    print(f"\nGenerating and saving forward volatility plot to {output_fwd_plot_filename}...")
    plot_forward_volatility(forward_vols_df, output_fwd_plot_filename)
    
    print("\n--- Applying PCA to Forward Volatilities ---")
    # Define the number of principal components for the demonstration
    num_pca_components = 3
    
    # Ensure num_components is not more than the number of features or samples
    effective_samples = forward_vols_df.iloc[1:].dropna(how='all').shape[0]
    effective_features = forward_vols_df.shape[1]

    if effective_samples == 0:
        print("No data available for PCA after cleaning. Skipping PCA plotting.")
    else:
        actual_n_components = min(num_pca_components, effective_samples, effective_features)
        if actual_n_components < num_pca_components:
            print(f"Warning: Requested {num_pca_components} PCA components, but only {actual_n_components} "
                  f"can be computed. Using n_components = {actual_n_components}.")
        else:
            print(f"Proceeding with {actual_n_components} PCA components.")

        if actual_n_components > 0:
            try:
                principal_components_df, pca_model = apply_pca_to_forward_vol(
                    forward_vols_df, 
                    n_components=actual_n_components
                )
                
                print("\nPrincipal Components DataFrame (head for plotting):")
                print(principal_components_df.head())
                
                print("\n--- Plotting Principal Components ---")
                output_pca_plot_filename = "pca_components_plot.png"
                print(f"\nGenerating and saving PCA components plot to {output_pca_plot_filename}...")
                plot_pca_components(principal_components_df, output_pca_plot_filename)

            except ValueError as ve:
                print(f"Error during PCA application for plotting: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred during PCA part for plotting: {e}")
        else:
            print("Cannot perform PCA with 0 components. Skipping PCA plotting.")

    print("\n--- Graphing script finished ---")
