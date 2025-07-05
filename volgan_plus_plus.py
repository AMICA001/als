import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. Synthetic Intraday Data Generation
def generate_mock_data(batch_size=64, surface_shape=(1, 32, 32), asset_dim=10):
    spot = torch.randn(batch_size, 1) * 30 + 4500         # Spot: ~4500 Â± 30
    vix = torch.rand(batch_size, 1) * 0.15 + 0.1          # VIX: 10%â€“25%
    realized_vol = torch.rand(batch_size, 1) * 0.1 + 0.2  # Realized vol
    state_t = torch.cat([spot, vix, realized_vol], dim=1)

    surface_t = torch.rand(batch_size, *surface_shape)
    skew = torch.linspace(1.3, 0.7, surface_shape[-2]).view(1, 1, -1, 1)
    surface_t1 = surface_t * skew + 0.01 * torch.randn_like(surface_t)  # Inject noise
    returns_t1 = torch.randn(batch_size, asset_dim) * 0.015
    prev_weights = F.softmax(torch.randn(batch_size, asset_dim), dim=1)
    target_vix = vix + 0.02 * torch.randn(batch_size, 1)

    # synthetic strike and maturity grids for Dupire/VIX calc
    strikes = torch.linspace(4000, 5000, surface_shape[-2]).repeat(batch_size, 1)
    maturities = torch.linspace(1/24, 1, surface_shape[-1]).repeat(batch_size, 1)

    return state_t, surface_t, surface_t1, returns_t1, prev_weights, target_vix, strikes, maturities

# 2. Model Definitions
class ConditionalForwardGenerator(nn.Module):
    def __init__(self, z_dim, state_dim, output_shape=(1, 32, 32)):
        super().__init__()
        self.fc = nn.Linear(z_dim + state_dim, 128)
        self.net = nn.Sequential(
            nn.Softplus(),
            nn.Linear(128, 512),
            nn.Softplus(),
            nn.Linear(512, int(torch.prod(torch.tensor(output_shape)))),
            nn.Softplus()
        )
        self.output_shape = output_shape

    def forward(self, z, state):
        x = torch.cat([z, state], dim=1)
        out = self.net(self.fc(x))
        # Scale the output of the final Softplus - this worked well previously
        out_scaled = out * 0.25
        return out_scaled.view(-1, *self.output_shape)

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
        self.surface_dim = int(torch.prod(torch.tensor(surface_shape)))
        self.model = nn.Sequential(
            nn.Linear(self.surface_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, surface):
        x = surface.view(surface.size(0), -1)
        return self.model(x)

# 3. Loss Functions
def sharpe_loss(weights, returns):
    portfolio_returns = torch.sum(weights * returns, dim=1)
    mean = torch.mean(portfolio_returns)
    std = torch.std(portfolio_returns) + 1e-6
    sharpe = mean / std
    return -sharpe.mean()

def constraint_loss(weights, prev_weights, delta_limit=0.1, max_weight_threshold=0.3, hhi_coeff=0.0):
    turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)
    turnover_penalty = F.relu(turnover - delta_limit).mean()

    max_weight_penalty = torch.sum(F.relu(weights - max_weight_threshold), dim=1).mean()

    # HHI = sum of squared weights. Higher HHI means more concentration.
    hhi_value = torch.sum(weights**2, dim=1).mean()
    hhi_penalty_value = hhi_value * hhi_coeff # Penalize HHI directly, scaled by hhi_coeff

    return turnover_penalty + max_weight_penalty + hhi_penalty_value

def arbitrage_penalty(surface, maturities_axis, strike_axis=None): # strike_axis not used yet but for consistency
    # Butterfly arbitrage (strike convexity)
    # surface shape: (batch, 1, N_strikes, N_maturities)
    dK = surface[:, :, 2:, :] - 2 * surface[:, :, 1:-1, :] + surface[:, :, :-2, :] # Check along strike axis (dim 2)
    penalty_butterfly = F.relu(-dK).mean()

    # Calendar spread arbitrage (non-decreasing total variance)
    # maturities_axis shape: (N_maturities,)
    variance_surface = surface**2
    # Ensure maturities_axis is correctly shaped for broadcasting: (1, 1, 1, N_maturities)
    maturities_row = maturities_axis.view(1, 1, 1, -1).to(surface.device)

    total_variance = variance_surface * maturities_row # (batch, 1, N_strikes, N_maturities)

    # Check along maturity axis (dim 3)
    # Diff is (V_j+1 * T_j+1) - (V_j * T_j)
    calendar_diff = total_variance[:, :, :, 1:] - total_variance[:, :, :, :-1]
    penalty_calendar = F.relu(-calendar_diff).mean()

    return penalty_butterfly + penalty_calendar # Combine penalties

def vix_replication_loss(surface, target_vix):
    # Approx VIXÂ² ~ avg of IVÂ² (mocked simplification)
    approx_vix_sq = torch.mean(surface**2, dim=[1,2,3])
    target_vix_sq = target_vix.view(-1)**2
    return ((approx_vix_sq - target_vix_sq)**2).mean()

def dupire_residual(surface, strikes, maturities):
    surface = surface.squeeze(1)
    dT = surface[:, :, 2:] - surface[:, :, :-2]
    dK2 = surface[:, 2:, :] - 2 * surface[:, 1:-1, :] + surface[:, :-2, :]

    # Normalize strikes
    strikes_normalized = strikes / 4500.0 # Using 4500 as the reference scale

    lhs = dT[:, 1:-1, :]
    # Use normalized strikes in the rhs calculation
    rhs = 0.5 * strikes_normalized[:, None, 1:-1]**2 * dK2[:, :, 1:-1]
    return ((lhs - rhs)**2).mean()

# 4. Training Loop
def train_one_epoch(G, D, P, optim_G, optim_D, lambda_arb=50.0, lambda_vix=1.0, lambda_dup=2e-4): # Adjusted dup lambda after strike norm
    state_t_raw, _, surface_t1, returns_t1, prev_weights, target_vix, strikes, maturities = generate_mock_data()

    # Normalize state_t components
    # spot (index 0) is around 4500, vix (index 1) is 0.1-0.25, realized_vol (index 2) is 0.2-0.3
    state_t_normalized = state_t_raw.clone()
    state_t_normalized[:, 0] = state_t_raw[:, 0] / 4500.0 # Normalize spot price
    # VIX and realized_vol are already in a smaller range, maybe no normalization needed or just centering.
    # For now, only normalizing spot.

    z = torch.randn(state_t_normalized.size(0), 16)
    # Use normalized state for G and D
    fake_surface = G(z, state_t_normalized)

    D_real = D(surface_t1, state_t_normalized) # Also use normalized state for Discriminator consistency
    D_fake = D(fake_surface.detach(), state_t_normalized)
    loss_D = -D_real.mean() + D_fake.mean()
    optim_D.zero_grad(); loss_D.backward(); optim_D.step()

    weights = P(fake_surface)
    D_fake_for_G = D(fake_surface, state_t_normalized) # Corrected variable name
    loss_gan = -D_fake_for_G.mean()
    loss_sharpe = sharpe_loss(weights, returns_t1)

    hhi_lambda_coeff = 0.5 # Coefficient for HHI penalty component
    loss_cons = constraint_loss(weights, prev_weights,
                                delta_limit=0.1,
                                max_weight_threshold=0.3,
                                hhi_coeff=hhi_lambda_coeff)

    # Prepare axes for arbitrage penalty
    # Assuming surface_cfg_shape is available or can be inferred
    # For now, re-create it based on common defaults, ideally pass from main or get from G.output_shape
    _surface_cfg_shape_temp = (1, 32, 32) # Matching default in generate_mock_data
    current_plot_maturities_axis = torch.linspace(1/24, 1, _surface_cfg_shape_temp[-1], device=fake_surface.device)
    # current_plot_strikes_axis = torch.linspace(4000, 5000, _surface_cfg_shape_temp[-2], device=fake_surface.device) # if needed

    loss_arb = arbitrage_penalty(fake_surface, current_plot_maturities_axis)
    loss_vix = vix_replication_loss(fake_surface, target_vix)
    loss_dup = dupire_residual(fake_surface, strikes, maturities)

    loss_G = loss_gan + loss_sharpe + loss_cons + lambda_arb*loss_arb + lambda_vix*loss_vix + lambda_dup*loss_dup
    optim_G.zero_grad(); loss_G.backward(); optim_G.step()

    return (loss_D.item(), loss_G.item(),
            loss_gan.item(), loss_sharpe.item(), loss_cons.item(),
            loss_arb.item(), loss_vix.item(), loss_dup.item(),
            fake_surface.mean().item(), target_vix.mean().item(),
            fake_surface[0].detach().cpu().numpy()) # Return first sample surface

# 5. Train and Plot
G = ConditionalForwardGenerator(z_dim=16, state_dim=3)
D = Discriminator(state_dim=3)
P = SharpeOptimizingPortfolio()

optim_G = torch.optim.Adam(list(G.parameters()) + list(P.parameters()), lr=1e-4)
optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)

losses_D, losses_G = [], []
# Store individual losses for plotting/analysis if needed
detailed_losses = {
    "gan": [], "sharpe": [], "cons": [], "arb": [], "vix": [], "dup": []
}

# Get sample strikes and maturities for plotting axes (assuming they are constant)
# This is a simplification; ideally, these should come with the sample_surface if they can vary per batch.
_, _, _, _, _, _, s_strikes_np, s_maturities_np = generate_mock_data(batch_size=1)
# Use meshgrid for 3D plot
s_strikes_np = s_strikes_np[0] # Take first batch item
s_maturities_np = s_maturities_np[0] # Take first batch item

# Correcting the shapes for meshgrid if they are (N,) and (M,)
# generate_mock_data returns strikes (batch, N) and maturities (batch, M)
# For plotting a single surface, we need one set of N strikes and M maturities.
# The surface itself is (num_strikes, num_maturities) after squeezing.
# Let's assume surface_shape = (1, num_strikes, num_maturities)
# So strikes will have num_strikes unique values, maturities num_maturities unique values.
# The current generate_mock_data makes strikes.shape[1] = surface_shape[-2] (num_strikes)
# and maturities.shape[1] = surface_shape[-1] (num_maturities)
# This is not quite right. Strikes and Maturities from generate_mock_data are grids.
# Let's redefine how we get them for plotting.
# We need unique strike values and unique maturity values.
surface_cfg_shape = (1, 32, 32) # Default from generate_mock_data
plot_strikes_axis = np.linspace(4000, 5000, surface_cfg_shape[-2])
plot_maturities_axis = np.linspace(1/24, 1, surface_cfg_shape[-1])
S_plot, M_plot = np.meshgrid(plot_strikes_axis, plot_maturities_axis)


plot_every_n_epochs = 10
fig_3d = None # To reuse figure window

for epoch in range(30): # Restore epochs
    ld, lg, l_gan, l_sharpe, l_cons, l_arb, l_vix, l_dup, mean_fake_surf, mean_target_vix, sample_surface_np = train_one_epoch(G, D, P, optim_G, optim_D)
    losses_D.append(ld); losses_G.append(lg)

    detailed_losses["gan"].append(l_gan)
    detailed_losses["sharpe"].append(l_sharpe)
    detailed_losses["cons"].append(l_cons)
    detailed_losses["arb"].append(l_arb)
    detailed_losses["vix"].append(l_vix)
    detailed_losses["dup"].append(l_dup)

    detailed_losses["dup"].append(l_dup)

    # These should ideally match the lambda values used in train_one_epoch
    # Read them from the function's defaults or pass them if they vary
    current_lambda_arb = 50.0
    current_lambda_vix = 1.0
    current_lambda_dup = 2e-4 # Updated to match train_one_epoch

    print(f"Epoch {epoch+1}: D loss = {ld:.4f}, G+P loss = {lg:.2e}")
    print(f"    Components: GAN={l_gan:.2e}, Sharpe={l_sharpe:.2e}, Cons={l_cons:.2e}")
    print(f"    Penalties: Arb={l_arb:.2e} (scaled: {current_lambda_arb*l_arb:.2e}), VIX={l_vix:.2e} (scaled: {current_lambda_vix*l_vix:.2e}), Dupire={l_dup:.2e} (scaled: {current_lambda_dup*l_dup:.2e})")
    print(f"    VIX Debug: Mean Fake Surface={mean_fake_surf:.4f}, Mean Target VIX={mean_target_vix:.4f}")

    if (epoch + 1) % plot_every_n_epochs == 0:
        if fig_3d is None:
            fig_3d = plt.figure(figsize=(10, 7))
        else:
            fig_3d.clf() # Clear previous plot

        ax = fig_3d.add_subplot(111, projection='3d')

        # sample_surface_np has shape (1, N_strikes, N_maturities) due to G.output_shape and selection of [0]
        # S_plot, M_plot from meshgrid(plot_strikes_axis, plot_maturities_axis)
        # S_plot shape: (N_maturities, N_strikes), M_plot shape: (N_maturities, N_strikes)
        # Surface data Z needs to match this. sample_surface_np[0] is (N_strikes, N_maturities)
        # So, we might need to transpose sample_surface_np[0] if S_plot, M_plot are (M,S) vs (S,M)

        Z_data = sample_surface_np.squeeze() # Shape (N_strikes, N_maturities)
        if Z_data.shape[0] == S_plot.shape[1] and Z_data.shape[1] == S_plot.shape[0]: # if Z is (S,M) and S_plot is (M,S)
             Z_data = Z_data.T # Transpose to (N_maturities, N_strikes)

        ax.plot_surface(S_plot, M_plot, Z_data, cmap='viridis')
        ax.set_xlabel('Strikes')
        ax.set_ylabel('Maturities')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'Generated Volatility Surface - Epoch {epoch+1}')
        plt.pause(0.1) # Pause to update the plot window

plt.figure(figsize=(12, 8)) # For the loss plots

plt.subplot(2, 1, 1)
plt.plot(losses_D, label="Discriminator Loss")
plt.plot(losses_G, label="Generator+Portfolio Loss (Total)")
plt.yscale('symlog') # Use symlog if losses are very large or negative
plt.legend(); plt.grid(); plt.title("Overall Training Losses")

plt.subplot(2, 1, 2)
plt.plot(detailed_losses["gan"], label="GAN Loss")
plt.plot(detailed_losses["sharpe"], label="Sharpe Loss")
plt.plot(detailed_losses["cons"], label="Constraint Loss")
plt.plot(detailed_losses["arb"], label="Arbitrage Penalty (unscaled)")
plt.plot(detailed_losses["vix"], label="VIX Penalty (unscaled)")
plt.plot(detailed_losses["dup"], label="Dupire Penalty (unscaled)")
plt.yscale('symlog')
plt.legend(); plt.grid(); plt.title("Individual G Loss Components (Unscaled)")

plt.tight_layout()
plt.show()
