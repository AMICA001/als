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
        return out.view(-1, *self.output_shape)

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
    lhs = dT[:, 1:-1, :]
    rhs = 0.5 * strikes[:, None, 1:-1]**2 * dK2[:, :, 1:-1]
    return ((lhs - rhs)**2).mean()

# 4. Training Loop
def train_one_epoch(G, D, P, optim_G, optim_D, lambda_arb=5.0, lambda_vix=5.0, lambda_dup=2.0):
    state_t, _, surface_t1, returns_t1, prev_weights, target_vix, strikes, maturities = generate_mock_data()

    z = torch.randn(state_t.size(0), 16)
    fake_surface = G(z, state_t)

    D_real = D(surface_t1, state_t)
    D_fake = D(fake_surface.detach(), state_t)
    loss_D = -D_real.mean() + D_fake.mean()
    optim_D.zero_grad(); loss_D.backward(); optim_D.step()

    weights = P(fake_surface)
    D_fake_for_G = D(fake_surface, state_t)
    loss_gan = -D_fake_for_G.mean()
    loss_sharpe = sharpe_loss(weights, returns_t1)
    loss_cons = constraint_loss(weights, prev_weights)
    loss_arb = arbitrage_penalty(fake_surface)
    loss_vix = vix_replication_loss(fake_surface, target_vix)
    loss_dup = dupire_residual(fake_surface, strikes, maturities)

    loss_G = loss_gan + loss_sharpe + loss_cons + lambda_arb*loss_arb + lambda_vix*loss_vix + lambda_dup*loss_dup
    optim_G.zero_grad(); loss_G.backward(); optim_G.step()

    return loss_D.item(), loss_G.item()

# 5. Train and Plot
G = ConditionalForwardGenerator(z_dim=16, state_dim=3)
D = Discriminator(state_dim=3)
P = SharpeOptimizingPortfolio()

optim_G = torch.optim.Adam(list(G.parameters()) + list(P.parameters()), lr=1e-4)
optim_D = torch.optim.Adam(D.parameters(), lr=1e-4)

losses_D, losses_G = [], []
for epoch in range(30):
    ld, lg = train_one_epoch(G, D, P, optim_G, optim_D)
    losses_D.append(ld); losses_G.append(lg)
    print(f"Epoch {epoch+1}: D loss = {ld:.4f}, G+P loss = {lg:.4f}")

plt.plot(losses_D, label="Discriminator Loss")
plt.plot(losses_G, label="Generator+Portfolio Loss")
plt.legend(); plt.grid(); plt.title("Training Loss"); plt.show()
