import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        # Scale down the output before reshaping and Softplus in self.net if Softplus is the last layer of self.net
        # The current self.net in __init__ already has Softplus as its last layer.
        # So, this scaling should ideally happen *inside* self.net, before its final Softplus.
        # For a quick test, if self.net was just a linear layer, we'd scale its output.
        # Given nn.Sequential, it's better to modify the last layer of self.net or add a scaling layer.

        # Simpler approach for now: if the last layer of self.net is Linear, then Softplus,
        # we can't easily intercept.
        # Let's assume self.net outputs logits, and we apply scaling + softplus here.
        # This requires changing self.net to NOT have Softplus as its last layer.

        # Temporary modification for testing:
        # Let's assume self.net outputs values that are *too large* for Softplus.
        # We will scale the output of self.net directly.
        # This might not be ideal if Softplus is important for non-linearity throughout self.net.

        # The last operation in self.net is nn.Softplus().
        # To scale *before* this, we'd have to redefine self.net or intercept.
        # A simpler, slightly hacky way for now is to scale the *output* of Softplus,
        # with the understanding this isn't the same as scaling *into* Softplus.
        # output_scaled = out * 0.25 # This scales *after* Softplus.

        # Correct approach: Modify self.net to not include the final Softplus, then apply here.
        # For now, let's try to make the network learn smaller values by other means (lambdas).
        # Reverting to explore lambdas first, as direct model modification is more involved.
        # The previous step was to increase lambda_vix. We saw a slight decrease in Mean Fake Surface.
        # Let's try a more aggressive lambda_vix.

        # Re-evaluation: The prompt implies I should try scaling the generator output.
        # The current ConditionalForwardGenerator.net already ends with Softplus.
        # To scale *before* this final Softplus, I need to modify the definition of self.net.
        # I will remove the last Softplus from self.net and apply scaling then Softplus in forward().

        raw_output = self.fc(x) # Output of the first linear layer
        # Pass through all but the last Softplus of the original net
        # Original net: Softplus, Linear, Softplus, Linear, Softplus
        # New net_before_final_softplus: Softplus, Linear, Softplus, Linear

        # This requires restructuring __init__ to make self.net more modular.
        # Let's try a simpler scaling first: scale the final output of G, then backprop
        # will hopefully adjust weights to produce smaller pre-Softplus values.
        # This is an indirect way.
        out_scaled = out * 0.25 # Scale the output of the final Softplus

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

def constraint_loss(weights, prev_weights, delta_limit=0.1):
    turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)
    turnover_penalty = F.relu(turnover - delta_limit).mean()
    greek_penalty = torch.sum(F.relu(weights - 0.3), dim=1).mean()
    return turnover_penalty + greek_penalty

def arbitrage_penalty(surface):
    dK = surface[:, :, 2:] - 2 * surface[:, :, 1:-1] + surface[:, :, :-2]
    return F.relu(-dK).mean()

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
    loss_cons = constraint_loss(weights, prev_weights)
    loss_arb = arbitrage_penalty(fake_surface)
    loss_vix = vix_replication_loss(fake_surface, target_vix)
    loss_dup = dupire_residual(fake_surface, strikes, maturities)

    loss_G = loss_gan + loss_sharpe + loss_cons + lambda_arb*loss_arb + lambda_vix*loss_vix + lambda_dup*loss_dup
    optim_G.zero_grad(); loss_G.backward(); optim_G.step()

    return (loss_D.item(), loss_G.item(),
            loss_gan.item(), loss_sharpe.item(), loss_cons.item(),
            loss_arb.item(), loss_vix.item(), loss_dup.item(),
            fake_surface.mean().item(), target_vix.mean().item())

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

for epoch in range(30): # Restore epochs
    ld, lg, l_gan, l_sharpe, l_cons, l_arb, l_vix, l_dup, mean_fake_surf, mean_target_vix = train_one_epoch(G, D, P, optim_G, optim_D)
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

    # For VIX loss debugging:
    # Need to get fake_surface from train_one_epoch or re-calculate for print if not returned
    # This part is tricky as train_one_epoch doesn't return fake_surface
    # For now, we'll skip printing fake_surface.mean() and target_vix.mean() here
    # and focus on the effect of lambda_dup.
    # If VIX loss is still an issue, we'll need to modify train_one_epoch to return them.

    print(f"Epoch {epoch+1}: D loss = {ld:.4f}, G+P loss = {lg:.2e}")
    print(f"    Components: GAN={l_gan:.2e}, Sharpe={l_sharpe:.2e}, Cons={l_cons:.2e}")
    print(f"    Penalties: Arb={l_arb:.2e} (scaled: {current_lambda_arb*l_arb:.2e}), VIX={l_vix:.2e} (scaled: {current_lambda_vix*l_vix:.2e}), Dupire={l_dup:.2e} (scaled: {current_lambda_dup*l_dup:.2e})")
    print(f"    VIX Debug: Mean Fake Surface={mean_fake_surf:.4f}, Mean Target VIX={mean_target_vix:.4f}")

plt.figure(figsize=(12, 8))

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
