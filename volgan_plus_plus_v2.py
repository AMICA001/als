import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Jump Diffusion Spot Simulation (Merton Model)
def simulate_spot_jump_diffusion(S0, mu=0.0, sigma=0.2, lamb=0.1, mu_j=0.0, sigma_j=0.02, dt=1/252, steps=1):
    batch_size = S0.shape[0]
    W = torch.randn(batch_size, steps) * np.sqrt(dt)
    N = torch.poisson(lamb * dt * torch.ones(batch_size, steps))
    jump = N * (mu_j + sigma_j * torch.randn(batch_size, steps))
    diffusion = (mu - 0.5 * sigma**2) * dt + sigma * W
    log_return = diffusion + jump
    log_S = torch.log(S0.unsqueeze(1)) + torch.cumsum(log_return, dim=1)
    return torch.exp(log_S[:, -1])

# 2. Surface Cleaning from Excel-style DataFrame
def clean_and_structure_surface(raw_surface_df):
    pivot = raw_surface_df.pivot(index='Strike', columns='Maturity', values='IV')
    pivot = pivot.sort_index().sort_index(axis=1)
    surface_tensor = torch.tensor(pivot.values, dtype=torch.float32).unsqueeze(0)  # (1, S, M)
    return surface_tensor

# 3. Data Generator with Spot Jump Diffusion
def generate_mock_data(batch_size=64, surface_shape=(1, 32, 32), asset_dim=10):
    spot = simulate_spot_jump_diffusion(torch.full((batch_size,), 4500.0))
    vix = torch.rand(batch_size, 1) * 0.15 + 0.1
    realized_vol = torch.rand(batch_size, 1) * 0.1 + 0.2
    state_t = torch.cat([spot.view(-1, 1), vix, realized_vol], dim=1)

    surface_t = torch.rand(batch_size, *surface_shape)
    skew = torch.linspace(1.3, 0.7, surface_shape[-2]).view(1, 1, -1, 1)
    surface_t1 = surface_t * skew + 0.01 * torch.randn_like(surface_t)
    returns_t1 = torch.randn(batch_size, asset_dim) * 0.015
    prev_weights = F.softmax(torch.randn(batch_size, asset_dim), dim=1)
    target_vix = vix + 0.02 * torch.randn(batch_size, 1)

    strikes = torch.linspace(4000, 5000, surface_shape[-2]).repeat(batch_size, 1)
    maturities = torch.linspace(1/24, 1, surface_shape[-1]).repeat(batch_size, 1)

    return state_t, surface_t, surface_t1, returns_t1, prev_weights, target_vix, strikes, maturities

# 4. Model Definitions
class ConditionalForwardGenerator(nn.Module):
    def __init__(self, z_dim, state_dim, output_shape=(1, 32, 32)):
        super().__init__()
        self.fc = nn.Linear(z_dim + state_dim, 128)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, int(torch.prod(torch.tensor(output_shape)))),
            nn.Softplus()
        )
        self.output_shape = output_shape

    def forward(self, z, state):
        x = torch.cat([z, state], dim=1)
        out = self.net(self.fc(x))
        return out.view(-1, *self.output_shape) * 0.25

class Discriminator(nn.Module):
    def __init__(self, state_dim, input_shape=(1, 32, 32)):
        super().__init__()
        self.input_dim = int(torch.prod(torch.tensor(input_shape))) + state_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, surface, state):
        flat = surface.view(surface.size(0), -1)
        x = torch.cat([flat, state], dim=1)
        return self.net(x)

class SharpeOptimizingPortfolio(nn.Module):
    def __init__(self, surface_shape=(1, 32, 32), output_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(surface_shape))), 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, surface):
        x = surface.view(surface.size(0), -1)
        return self.model(x)

# 5. Losses
def sharpe_loss(weights, returns):
    portfolio_returns = torch.sum(weights * returns, dim=1)
    return -torch.mean(portfolio_returns / (portfolio_returns.std() + 1e-6))

def constraint_loss(weights, prev_weights, delta_limit=0.1, max_weight_threshold=0.3, hhi_coeff=0.0):
    turnover = torch.sum(torch.abs(weights - prev_weights), dim=1).mean()
    max_penalty = torch.sum(F.relu(weights - max_weight_threshold), dim=1).mean()
    hhi = torch.sum(weights**2, dim=1).mean() * hhi_coeff
    return F.relu(turnover - delta_limit) + max_penalty + hhi

def arbitrage_penalty(surface, maturities_axis):
    dK = surface[:, :, 2:, :] - 2 * surface[:, :, 1:-1, :] + surface[:, :, :-2, :]
    butterfly = F.relu(-dK).mean()
    T = maturities_axis.view(1, 1, 1, -1).to(surface.device)
    total_var = surface**2 * T
    calendar = F.relu(-(total_var[:, :, :, 1:] - total_var[:, :, :, :-1])).mean()
    return butterfly + calendar

def vix_replication_loss(surface, target_vix):
    approx_vix_sq = torch.mean(surface**2, dim=[1, 2, 3])
    return ((approx_vix_sq - target_vix.view(-1)**2)**2).mean()

# This is the simpler, correct version of dupire_residual
def dupire_residual(surface, strikes, maturities):
    surface = surface.squeeze(1) # surface is (B, S, M)
    dT = surface[:, :, 2:] - surface[:, :, :-2]
    dK2 = surface[:, 2:, :] - 2 * surface[:, 1:-1, :] + surface[:, :-2, :]
    K = strikes / 4500.0 # K is (B,S)

    lhs = dT[:, 1:-1, :] # Sliced to (B, S-2, M-2)

    # K[:, None, 1:-1] gives (B, 1, S-2). K[:, 1:-1, None] would be (B, S-2, 1)
    # dK2[:, :, 1:-1] is (B, S-2, M-2)
    rhs = 0.5 * K[:, None, 1:-1]**2 * dK2[:, :, 1:-1]
    return ((lhs - rhs)**2).mean()

def surface_smoothness_penalty(surface):
    dK = surface[:, :, 1:, :] - surface[:, :, :-1, :]
    dT = surface[:, :, :, 1:] - surface[:, :, :, :-1]
    return (dK**2).mean() + (dT**2).mean()

def gradient_penalty(D, real, fake, state):
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_()
    d_interpolated = D(interp, state)
    grads = torch.autograd.grad(d_interpolated, interp,
                                torch.ones_like(d_interpolated), create_graph=True)[0]
    return ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1)**2).mean()

# === New Helper Functions for Forward Risk Constraint ===
def get_next_period_portfolio_return_distribution(current_state_norm_sample, G, P,
                                                 num_z_samples, asset_dim, base_return_std_dev, device):
    """
    Generates a distribution of next-period portfolio returns for a given state.
    G and P are the generator and portfolio models.
    current_state_norm_sample should be a single sample (e.g., [state_dim]).
    """
    portfolio_returns = []

    # Ensure current_state_norm_sample is correctly shaped (1, state_dim) for G
    if current_state_norm_sample.ndim == 1:
        current_state_norm_sample = current_state_norm_sample.unsqueeze(0)

    for _ in range(num_z_samples):
        z_g = torch.randn(1, 16, device=device) # Assuming z_dim = 16, hardcoded for now

        # Important: For gradient flow, G and P should not be in eval() mode here
        # if this function is called within train_one_epoch and meant to affect G/P grads.
        # However, for stability, often such simulations are done with models in eval mode
        # or by detaching parts of the computation if only penalizing current weights.
        # For now, assume G and P are in the mode set by train_one_epoch.

        next_surface = G(z_g, current_state_norm_sample)
        next_weights = P(next_surface) # next_weights shape (1, asset_dim)

        # Simulate asset returns for the next period
        # asset_returns shape should be (1, asset_dim) for batch size 1
        asset_returns = torch.randn(1, asset_dim, device=device) * base_return_std_dev

        current_portfolio_return = torch.sum(next_weights.squeeze(0) * asset_returns.squeeze(0)) # Ensure 1D for sum
        portfolio_returns.append(current_portfolio_return)

    return torch.stack(portfolio_returns)

def calculate_var_loss(portfolio_returns_dist, confidence_level=0.95, var_limit=0.02):
    """
    Calculates VaR and the loss if VaR exceeds var_limit.
    VaR is calculated as a positive value representing potential loss.
    """
    if not isinstance(portfolio_returns_dist, torch.Tensor):
        portfolio_returns_dist = torch.tensor(portfolio_returns_dist)

    # VaR: (1-confidence_level) quantile of losses, or -quantile of returns
    # Ensure it's a loss (positive if return is negative)
    var_value = -torch.quantile(portfolio_returns_dist, 1.0 - confidence_level)

    loss = F.relu(var_value - var_limit)
    return loss, var_value.item() # Return raw VaR too for logging

# 6. Training Loop
# Added new lambdas and config for forward risk
def train_one_epoch(G, D, P, optim_G, optim_D,
                    lambda_arb=50.0, lambda_vix=1.0, lambda_dup=2e-4,
                    lambda_smooth=1e-3, lambda_gp=10.0,
                    lambda_fwd_risk=1.0, # New lambda for forward risk
                    num_z_samples_for_risk=50, # Num samples for risk dist
                    var_confidence_level=0.95,
                    var_limit=0.02, # Max 2% VaR
                    asset_dim=10, # Passed for asset returns generation
                    base_return_std_dev=0.015): # Passed for asset returns generation

    state_t, _, surface_t1, returns_t1, prev_weights, target_vix, strikes, maturities = generate_mock_data(asset_dim=asset_dim)

    state_t_norm = state_t.clone()
    state_t_norm[:, 0] = state_t[:, 0] / 4500.0  # normalize spot

    z = torch.randn(state_t.size(0), 16) # Assuming z_dim = 16
    fake_surface = G(z, state_t_norm)

    # Discriminator update with gradient penalty
    D_real = D(surface_t1, state_t_norm)
    D_fake = D(fake_surface.detach(), state_t_norm)
    gp = gradient_penalty(D, surface_t1, fake_surface.detach(), state_t_norm)
    loss_D = -D_real.mean() + D_fake.mean() + lambda_gp * gp
    optim_D.zero_grad(); loss_D.backward(); optim_D.step()

    # Generator update
    weights = P(fake_surface)
    D_fake_for_G = D(fake_surface, state_t_norm)

    loss_gan = -D_fake_for_G.mean()
    loss_sharpe = sharpe_loss(weights, returns_t1)
    loss_cons = constraint_loss(weights, prev_weights, hhi_coeff=0.5)

    current_maturities_axis = torch.linspace(1/24, 1, fake_surface.shape[-1], device=fake_surface.device)

    loss_arb = arbitrage_penalty(fake_surface, current_maturities_axis)
    loss_vix = vix_replication_loss(fake_surface, target_vix)
    loss_dup = dupire_residual(fake_surface, strikes, maturities)
    loss_smooth = surface_smoothness_penalty(fake_surface)
    loss_temporal = ((fake_surface - surface_t1)**2).mean()

    # === Forward Risk Calculation ===
    # Select first sample from batch for this calculation to keep it simple
    # G and P are in train mode here, so gradients will flow back from this risk loss
    # Note: prev_weights[0] is not used by get_next_period_portfolio_return_distribution
    # as it calculates next_weights internally based on G and P.
    # The function expects a single state sample, not a batch.
    portfolio_return_dist = get_next_period_portfolio_return_distribution(
        current_state_norm_sample=state_t_norm[0], # Single sample
        G=G, P=P,
        num_z_samples=num_z_samples_for_risk,
        asset_dim=asset_dim, # Need asset_dim from G's portfolio P
        base_return_std_dev=base_return_std_dev, # Std dev for mock returns
        device=fake_surface.device
    )
    loss_fwd_risk, raw_var_value = calculate_var_loss(
        portfolio_return_dist,
        confidence_level=var_confidence_level,
        var_limit=var_limit
    )

    loss_G = (loss_gan + loss_sharpe + loss_cons +
              lambda_arb * loss_arb + lambda_vix * loss_vix +
              lambda_dup * loss_dup + lambda_smooth * loss_smooth +
              0.5 * loss_temporal +
              lambda_fwd_risk * loss_fwd_risk) # Added forward risk loss

    optim_G.zero_grad(); loss_G.backward(); optim_G.step()

    # Diagnostic prints for specific losses
    # print(f"    Raw VIX Loss: {loss_vix.item():.6e}, Raw Dupire Loss: {loss_dup.item():.6e}")

    return (loss_D.item(), loss_G.item(), loss_gan.item(), loss_sharpe.item(), loss_cons.item(),
            loss_arb.item(), loss_vix.item(), loss_dup.item(), loss_smooth.item(), loss_temporal.item(),
            loss_fwd_risk.item(), raw_var_value, # Added forward risk loss and raw VaR
            fake_surface[0].detach().cpu().numpy(),
            surface_t1[0].detach().cpu().numpy(),
            target_vix.mean().item(), fake_surface.mean().item())

# Helper function for OOS evaluation
def get_atm_3m_iv(surface_tensor, spot_price, strikes_grid_for_sample, maturities_grid_for_sample):
    """
    Extracts At-The-Money (ATM) 3-Month implied volatility from a surface.

    Args:
        surface_tensor (torch.Tensor): The volatility surface (shape [1, S, M] or [S, M]).
        spot_price (float or torch.Tensor): The spot price for this surface.
        strikes_grid_for_sample (torch.Tensor): 1D tensor of strike prices for this surface.
        maturities_grid_for_sample (torch.Tensor): 1D tensor of maturity values (in years) for this surface.

    Returns:
        float: The ATM 3-Month IV. Returns NaN if not found.
    """
    if surface_tensor.ndim == 3 and surface_tensor.shape[0] == 1:
        surface_tensor = surface_tensor.squeeze(0) # Shape [S, M]

    if surface_tensor.shape[0] != strikes_grid_for_sample.shape[0] or \
       surface_tensor.shape[1] != maturities_grid_for_sample.shape[0]:
        # Try transposing the surface if dimensions seem swapped
        if surface_tensor.shape[1] == strikes_grid_for_sample.shape[0] and \
           surface_tensor.shape[0] == maturities_grid_for_sample.shape[0]:
            surface_tensor = surface_tensor.T
        else:
            # This case indicates a potential mismatch that needs debugging if it occurs.
            # For now, returning NaN or raising error.
            # print(f"Warning: Shape mismatch. Surface: {surface_tensor.shape}, Strikes: {strikes_grid_for_sample.shape}, Mats: {maturities_grid_for_sample.shape}")
            return np.nan


    # Find index for 3-month maturity (approx 0.25 years)
    target_maturity = 0.25
    maturity_idx = torch.argmin(torch.abs(maturities_grid_for_sample - target_maturity)).item()

    # Find index for ATM strike (strike closest to spot_price)
    atm_strike_idx = torch.argmin(torch.abs(strikes_grid_for_sample - spot_price)).item()

    iv = surface_tensor[atm_strike_idx, maturity_idx].item()
    return iv

# 7. Training Setup and Execution
G = ConditionalForwardGenerator(z_dim=16, state_dim=3, output_shape=(1, 32, 32))
D = Discriminator(state_dim=3, input_shape=(1, 32, 32))
P = SharpeOptimizingPortfolio(surface_shape=(1, 32, 32), output_dim=10)

optim_G = torch.optim.Adam(list(G.parameters()) + list(P.parameters()), lr=1e-4)
optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)

losses_D, losses_G = [], []
components = {"gan": [], "sharpe": [], "cons": [], "arb": [], "vix": [], "dup": [], "smooth": [], "temp": []}

plot_strikes_axis = np.linspace(4000, 5000, 32)
plot_maturities_axis = np.linspace(1/24, 1, 32)
S_grid, M_grid = np.meshgrid(plot_strikes_axis, plot_maturities_axis)
plot_every_n_epochs = 10
fig_3d_surf = None

for epoch in range(30): # Restored epochs
    epoch_outputs = train_one_epoch(G, D, P, optim_G, optim_D)
    # Added sample_target_surf_np and mean_fake_surf
    ld, lg, l_gan, l_sharpe, l_cons, l_arb, l_vix, l_dup, l_smooth, l_temp, \
    sample_fake_surf_np, sample_target_surf_np, tvix_mean, mean_fake_surf = epoch_outputs

    losses_D.append(ld); losses_G.append(lg)
    components["gan"].append(l_gan); components["sharpe"].append(l_sharpe)
    components["cons"].append(l_cons); components["arb"].append(l_arb)
    components["vix"].append(l_vix); components["dup"].append(l_dup)
    components["smooth"].append(l_smooth); components["temp"].append(l_temp)

    print(f"Epoch {epoch+1:02d} | D: {ld:.3f} | G: {lg:.3f} | GAN: {l_gan:.3f} | Sharpe: {l_sharpe:.3f} | "
          f"Cons: {l_cons:.3f} | Arb: {l_arb:.3f} | VIX: {l_vix:.3f} (Raw: {l_vix:.6e}) | Dup: {l_dup:.3f} (Raw: {l_dup:.6e}) | Smooth: {l_smooth:.3f} | Temp: {l_temp:.3f}")
    print(f"    VIX Debug: Mean Fake Surface={mean_fake_surf:.4f}, Mean Target VIX={tvix_mean:.4f}")


    if (epoch + 1) % plot_every_n_epochs == 0:
        if fig_3d_surf is None:
            fig_3d_surf = plt.figure(f"Volatility Surfaces (Epoch {epoch+1})", figsize=(16, 7)) # Adjusted figsize
        else:
            fig_3d_surf.clf()
            fig_3d_surf.suptitle(f"Volatility Surfaces (Epoch {epoch+1})")

        # Subplot 1: Generated (Fake) Surface
        ax1 = fig_3d_surf.add_subplot(121, projection='3d')
        Z_fake = sample_fake_surf_np.squeeze() # Renamed from sample_surf_np for clarity
        if Z_fake.shape[0] == S_grid.shape[1] and Z_fake.shape[1] == S_grid.shape[0]:
             Z_fake = Z_fake.T
        ax1.plot_surface(S_grid, M_grid, Z_fake, cmap='viridis')
        ax1.set_xlabel("Strike"); ax1.set_ylabel("Maturity"); ax1.set_zlabel("IV")
        ax1.set_title("Generated (Fake) Surface")

        # Subplot 2: Target (Simulated) Surface
        ax2 = fig_3d_surf.add_subplot(122, projection='3d')
        Z_target = sample_target_surf_np.squeeze()
        if Z_target.shape[0] == S_grid.shape[1] and Z_target.shape[1] == S_grid.shape[0]:
             Z_target = Z_target.T
        ax2.plot_surface(S_grid, M_grid, Z_target, cmap='magma') # Different cmap
        ax2.set_xlabel("Strike"); ax2.set_ylabel("Maturity"); ax2.set_zlabel("IV")
        ax2.set_title("Target (Simulated) Surface")

        plt.pause(0.1) # Allow plot to update

plt.figure("Training Losses", figsize=(12, 10)) # Keep this figure separate for loss plots

plt.subplot(2,1,1)
plt.plot(losses_D, label="Discriminator Loss")
plt.plot(losses_G, label="Total Generator Loss")
plt.yscale('symlog')
plt.title("Overall Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss (symlog scale)")
plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
for key, val_list in components.items():
    plt.plot(val_list, label=key)
plt.yscale('symlog')
plt.title("Generator Loss Components (Unscaled)")
plt.xlabel("Epoch")
plt.ylabel("Loss Component Value (symlog scale)")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0)); plt.grid(True)

plt.tight_layout()
plt.show()
print("Training complete. Close plot windows to proceed to OOS evaluation.")

# 9. Post-Training Out-of-Sample (OOS) Evaluation
print("\nStarting Out-of-Sample Evaluation...")
G.eval() # Set generator to evaluation mode

oos_samples = 100
predicted_atm_3m_ivs = []
actual_atm_3m_ivs = []

# Get the unique strike and maturity axes configurations, assuming they are fixed by surface_shape
# These are the same as plot_strikes_axis and plot_maturities_axis if surface_shape is constant
surface_cfg_shape_oos = (1, 32, 32) # Should match G's output_shape dimensions used in training
oos_strike_axis = torch.linspace(4000, 5000, surface_cfg_shape_oos[-2])
oos_maturity_axis = torch.linspace(1/24, 1, surface_cfg_shape_oos[-1])


for i in range(oos_samples):
    # Generate a single OOS sample
    state_t, _, surface_t1, _, _, _, strikes_full_grid, maturities_full_grid = \
        generate_mock_data(batch_size=1, surface_shape=surface_cfg_shape_oos[1:]) # Use the inner dims

    current_spot = state_t[0, 0].item() # Spot price for this sample

    state_t_norm = state_t.clone()
    state_t_norm[:, 0] = state_t_norm[:, 0] / 4500.0

    z_oos = torch.randn(1, 16) # Assuming z_dim = 16

    with torch.no_grad(): # No need to track gradients for OOS evaluation
        fake_surface_oos = G(z_oos, state_t_norm)

    # Extract ATM 3M IVs
    # fake_surface_oos is (1, 1, S, M), surface_t1 is (1, 1, S, M)
    # get_atm_3m_iv expects (S,M) or (1,S,M) for surface, and 1D strikes/maturities

    pred_iv = get_atm_3m_iv(fake_surface_oos.squeeze(0), current_spot, oos_strike_axis, oos_maturity_axis)
    act_iv = get_atm_3m_iv(surface_t1.squeeze(0), current_spot, oos_strike_axis, oos_maturity_axis)

    if not (np.isnan(pred_iv) or np.isnan(act_iv)):
        predicted_atm_3m_ivs.append(pred_iv)
        actual_atm_3m_ivs.append(act_iv)

    if (i + 1) % 10 == 0:
        print(f"Processed OOS sample {i+1}/{oos_samples}")

# Convert lists to numpy arrays for plotting and stats
predicted_atm_3m_ivs_np = np.array(predicted_atm_3m_ivs)
actual_atm_3m_ivs_np = np.array(actual_atm_3m_ivs)

# Step 4: Plot OOS Time Series (Covered in next step by plan)
# Step 5: Calculate and Print OOS Statistics (Covered in next step by plan)

print("OOS Evaluation Done.")

# 4. Plot OOS Time Series
if len(actual_atm_3m_ivs_np) > 0 and len(predicted_atm_3m_ivs_np) > 0:
    plt.figure("OOS ATM 3M IV Prediction", figsize=(12, 6))
    plt.plot(actual_atm_3m_ivs_np, label='Actual ATM 3M IV', marker='o', linestyle='-')
    plt.plot(predicted_atm_3m_ivs_np, label='Predicted ATM 3M IV', marker='x', linestyle='--')
    plt.title('Out-of-Sample: Actual vs. Predicted ATM 3-Month IV')
    plt.xlabel('OOS Sample Index')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.show() # Show this plot separately

    # 5. Calculate and Print OOS Statistics
    mae = np.mean(np.abs(actual_atm_3m_ivs_np - predicted_atm_3m_ivs_np))
    rmse = np.sqrt(np.mean((actual_atm_3m_ivs_np - predicted_atm_3m_ivs_np)**2))
    # MAPE - careful with actuals being zero, though IVs shouldn't be zero. Add epsilon for safety.
    mape = np.mean(np.abs((actual_atm_3m_ivs_np - predicted_atm_3m_ivs_np) / (actual_atm_3m_ivs_np + 1e-8))) * 100

    print("\nOOS ATM 3M IV Statistics:")
    print(f"  Number of OOS samples: {len(actual_atm_3m_ivs_np)}")
    print(f"  Mean Actual IV: {np.mean(actual_atm_3m_ivs_np):.4f}")
    print(f"  Mean Predicted IV: {np.mean(predicted_atm_3m_ivs_np):.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Correlation
    if len(actual_atm_3m_ivs_np) > 1: # Need at least 2 points for correlation
        correlation_matrix = np.corrcoef(actual_atm_3m_ivs_np, predicted_atm_3m_ivs_np)
        correlation = correlation_matrix[0, 1]
        print(f"  Correlation: {correlation:.4f}")
        print(f"  R-squared: {correlation**2:.4f}")

else:
    print("No valid OOS IVs collected to plot or calculate stats.")

print("\nEnd of OOS Evaluation.")
