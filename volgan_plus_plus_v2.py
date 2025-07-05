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

# 6. Training Loop
def train_one_epoch(G, D, P, optim_G, optim_D, lambda_arb=50.0, lambda_vix=1.0, lambda_dup=2e-4, lambda_smooth=1e-3, lambda_gp=10.0): # Reverted lambda_dup
    state_t, _, surface_t1, returns_t1, prev_weights, target_vix, strikes, maturities = generate_mock_data()

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

    loss_G = (loss_gan + loss_sharpe + loss_cons +
              lambda_arb * loss_arb + lambda_vix * loss_vix +
              lambda_dup * loss_dup + lambda_smooth * loss_smooth +
              0.5 * loss_temporal)

    optim_G.zero_grad(); loss_G.backward(); optim_G.step()

    # Diagnostic prints for specific losses
    # print(f"    Raw VIX Loss: {loss_vix.item():.6e}, Raw Dupire Loss: {loss_dup.item():.6e}")

    return (loss_D.item(), loss_G.item(), loss_gan.item(), loss_sharpe.item(), loss_cons.item(),
            loss_arb.item(), loss_vix.item(), loss_dup.item(), loss_smooth.item(), loss_temporal.item(),
            fake_surface[0].detach().cpu().numpy(),
            target_vix.mean().item(), fake_surface.mean().item()) # Added fake_surface.mean()

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
    # Added mean_fake_surf from return values
    ld, lg, l_gan, l_sharpe, l_cons, l_arb, l_vix, l_dup, l_smooth, l_temp, sample_surf_np, tvix_mean, mean_fake_surf = epoch_outputs

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
            fig_3d_surf = plt.figure(f"Generated Surface (Epoch {epoch+1})", figsize=(10, 7))
        else:
            fig_3d_surf.clf()
            fig_3d_surf.suptitle(f"Generated Surface (Epoch {epoch+1})")

        ax = fig_3d_surf.add_subplot(111, projection='3d')
        Z = sample_surf_np.squeeze()

        if Z.shape[0] == S_grid.shape[1] and Z.shape[1] == S_grid.shape[0]:
             Z = Z.T

        ax.plot_surface(S_grid, M_grid, Z, cmap='viridis')
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturity")
        ax.set_zlabel("Implied Volatility")
        plt.pause(0.1)

plt.figure("Training Losses", figsize=(12, 10))

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
print("Training complete. Close plot windows to exit.")
