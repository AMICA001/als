import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === 1. VAE for Volatility Surface Compression ===
class VolSurfaceVAE(nn.Module):
    def __init__(self, surface_shape=(1, 32, 32), latent_dim=8):
        super().__init__()
        self.surface_shape = surface_shape
        flat_dim = int(torch.prod(torch.tensor(surface_shape)))
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, flat_dim),
            nn.Softplus()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, *self.surface_shape)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def vae_loss(self, recon_x, x, mu, logvar, kl_weight=1.0): # Added kl_weight to signature
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_div

# === 2. GAN Generator for Latent Codes ===
class LatentGenerator(nn.Module):
    def __init__(self, z_dim, state_dim, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z, state): # z is GAN noise
        x = torch.cat([z, state], dim=1)
        return self.net(x)

# === 3. Discriminator on Latent Space ===
class LatentDiscriminator(nn.Module):
    def __init__(self, state_dim, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z_latent, state):
        x = torch.cat([z_latent, state], dim=1)
        return self.net(x)

# === 4. Portfolio Optimizer (Top-10 Options Only) ===
class TopKOptionPortfolio(nn.Module):
    def __init__(self, surface_shape=(1, 32, 32), top_k=10):
        super().__init__()
        self.surface_shape = surface_shape
        self.flat_dim = int(torch.prod(torch.tensor(surface_shape)))
        self.k = top_k
        self.linear = nn.Linear(self.flat_dim, self.flat_dim)

    def forward(self, surface):
        batch_size = surface.size(0)
        x = surface.view(batch_size, -1)
        raw_scores = self.linear(x)

        actual_k = min(self.k, raw_scores.size(1))
        _topk_vals, topk_idx = torch.topk(raw_scores, actual_k, dim=1)

        mask = torch.zeros_like(raw_scores)
        mask.scatter_(1, topk_idx, 1.0)

        masked_scores = raw_scores * mask + (1 - mask) * -1e9
        weights_flat = F.softmax(masked_scores, dim=1)

        final_weights_flat = weights_flat * mask

        return final_weights_flat.view(batch_size, *self.surface_shape)


# === 5. Utility Functions ===
def sharpe_loss_surface(weights, returns_surface):
    port_ret = (weights * returns_surface).sum(dim=[1,2,3])
    mean_return = torch.mean(port_ret)
    std_return = torch.std(port_ret) + 1e-6
    sharpe_ratio = mean_return / std_return
    return -sharpe_ratio

def constraint_loss(weights, max_weight=0.3):
    over_limit = F.relu(weights - max_weight)
    return over_limit.sum() / weights.size(0)

def gradient_penalty(D, real, fake, state):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, device=real.device)
    eps = eps.expand_as(real)

    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)

    d_interpolated = D(interp, state)

    grads = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads_flat = grads.view(batch_size, -1)
    grad_norm = grads_flat.norm(2, dim=1)
    gp = ((grad_norm - 1)**2).mean()
    return gp

# === 6. Sample Data Generator (Mock) ===
def generate_mock_surface(batch_size=64, surface_shape=(1, 32, 32)): # Matches user's last paste
    surface = torch.rand(batch_size, *surface_shape) * 0.3 + 0.05
    returns = torch.randn(batch_size, *surface_shape) * 0.01
    spot_raw = torch.full((batch_size, 1), 4500.0) + torch.randn(batch_size, 1) * 30 # Renamed to spot_raw
    vix = torch.rand(batch_size, 1) * 0.15 + 0.1
    realized = torch.rand(batch_size, 1) * 0.1 + 0.2
    state = torch.cat([spot_raw / 4500.0, vix, realized], dim=1) # Use spot_raw for state normalization
    return surface, returns, state, spot_raw # Return spot_raw

# Helper function for OOS evaluation (adapted from volgan_plus_plus_v2.py)
def get_atm_3m_iv(surface_tensor, spot_price, strikes_axis, maturities_axis):
    """
    Extracts At-The-Money (ATM) 3-Month implied volatility from a surface.
    Assumes surface_tensor is (C, S, M) or (S, M) after squeeze if needed.
    strikes_axis is 1D (S_dim), maturities_axis is 1D (M_dim).
    """
    if surface_tensor.ndim == 3: # e.g. (1, S, M)
        surface_tensor = surface_tensor.squeeze(0) # Shape [S, M]

    # Ensure surface_tensor is (S, M) matching strikes_axis (S) and maturities_axis (M)
    # Note: PyTorch default for meshgrid(X,Y) makes X (M,S) and Y (M,S) if X is S-dim, Y is M-dim.
    # Here, we assume surface_tensor is already (S_dim, M_dim)
    # And strikes_axis is (S_dim), maturities_axis is (M_dim)

    # Find index for 3-month maturity (approx 0.25 years)
    target_maturity = 0.25
    # Convert axes to tensors if they are numpy arrays, move to device of surface
    if isinstance(maturities_axis, np.ndarray):
        maturities_axis = torch.from_numpy(maturities_axis).to(surface_tensor.device).float()
    if isinstance(strikes_axis, np.ndarray):
        strikes_axis = torch.from_numpy(strikes_axis).to(surface_tensor.device).float()

    maturity_idx = torch.argmin(torch.abs(maturities_axis - target_maturity)).item()

    # Find index for ATM strike (strike closest to spot_price)
    atm_strike_idx = torch.argmin(torch.abs(strikes_axis - spot_price)).item()

    try:
        iv = surface_tensor[atm_strike_idx, maturity_idx].item()
    except IndexError:
        # This can happen if surface_tensor was (M,S) and we expected (S,M)
        # print(f"IndexError in get_atm_3m_iv. Surface: {surface_tensor.shape}, ATM idx: {atm_strike_idx}, Mat idx: {maturity_idx}. Trying transpose.")
        try:
            iv = surface_tensor.T[atm_strike_idx, maturity_idx].item()
        except Exception as e:
            # print(f"Error after transpose: {e}")
            return np.nan # Give up if transpose also fails
    return iv

# === Main Execution and Training Loops ===
if __name__ == '__main__':
    # Configuration
    SURFACE_SHAPE_CONFIG = (1, 32, 32)
    VAE_LATENT_DIM = 8
    GAN_Z_DIM = 16
    STATE_DIM = 3
    BATCH_SIZE = 64

    VAE_EPOCHS = 5
    VAE_LR = 1e-3
    VAE_KL_WEIGHT = 0.001

    LATENT_GAN_EPOCHS = 5
    LATENT_GAN_LR = 1e-4
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10.0

    PORTFOLIO_EPOCHS = 5
    PORTFOLIO_LR = 1e-3
    PORTFOLIO_MAX_WEIGHT = 0.3
    PORTFOLIO_CONSTRAINT_LAMBDA = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. VAE Pre-training ---
    print("\n--- Starting VAE Pre-training ---")
    vae = VolSurfaceVAE(surface_shape=SURFACE_SHAPE_CONFIG, latent_dim=VAE_LATENT_DIM).to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=VAE_LR)

    for epoch in range(VAE_EPOCHS):
        vae.train()
        original_surfaces, _, _, _ = generate_mock_surface( # Now unpacks 4
            batch_size=BATCH_SIZE,
            surface_shape=SURFACE_SHAPE_CONFIG
        )
        original_surfaces = original_surfaces.to(device)

        optimizer_vae.zero_grad()
        recon_surfaces, mu, logvar = vae(original_surfaces)
        loss = vae.vae_loss(recon_surfaces, original_surfaces, mu, logvar, kl_weight=VAE_KL_WEIGHT)
        loss.backward()
        optimizer_vae.step()

        if (epoch + 1) % 10 == 0 or epoch == VAE_EPOCHS -1:
            recon_mse_loss = F.mse_loss(recon_surfaces, original_surfaces, reduction='mean').item()
            kl_actual = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
            print(f"VAE Epoch [{epoch+1}/{VAE_EPOCHS}], Total Loss: {loss.item():.4f}, "
                  f"Recon MSE: {recon_mse_loss:.4f}, KL Div (mean): {kl_actual:.4f}")

    print("--- VAE Pre-training Finished ---")

    # --- 2. Latent GAN Training ---
    print("\n--- Starting Latent GAN Training ---")
    vae.eval()

    g_latent = LatentGenerator(z_dim=GAN_Z_DIM, state_dim=STATE_DIM, latent_dim=VAE_LATENT_DIM).to(device)
    d_latent = LatentDiscriminator(state_dim=STATE_DIM, latent_dim=VAE_LATENT_DIM).to(device)

    optimizer_g_latent = torch.optim.Adam(g_latent.parameters(), lr=LATENT_GAN_LR, betas=(0.5, 0.9))
    optimizer_d_latent = torch.optim.Adam(d_latent.parameters(), lr=LATENT_GAN_LR, betas=(0.5, 0.9))

    for epoch in range(LATENT_GAN_EPOCHS):
        for _ in range(CRITIC_ITERATIONS):
            d_latent.train()
            g_latent.eval()

            real_surfaces_for_d, _, state_for_d, _ = generate_mock_surface( # Unpack 4
                batch_size=BATCH_SIZE,
                surface_shape=SURFACE_SHAPE_CONFIG
            )
            real_surfaces_for_d = real_surfaces_for_d.to(device)
            state_for_d = state_for_d.to(device)

            z_noise_for_g_fake = torch.randn(BATCH_SIZE, GAN_Z_DIM, device=device)

            with torch.no_grad():
                mu_real, logvar_real = vae.encode(real_surfaces_for_d)
                real_latent_codes = vae.reparameterize(mu_real, logvar_real)

            fake_latent_codes_detached = g_latent(z_noise_for_g_fake, state_for_d).detach()

            optimizer_d_latent.zero_grad()

            d_real_score = d_latent(real_latent_codes, state_for_d).mean()
            d_fake_score = d_latent(fake_latent_codes_detached, state_for_d).mean()

            gp = gradient_penalty(d_latent, real_latent_codes, fake_latent_codes_detached, state_for_d)

            d_loss = d_fake_score - d_real_score + LAMBDA_GP * gp
            d_loss.backward()
            optimizer_d_latent.step()

        g_latent.train()
        d_latent.eval()

        _, _, state_for_g, _ = generate_mock_surface( # Unpack 4
            batch_size=BATCH_SIZE, surface_shape=SURFACE_SHAPE_CONFIG
        )
        state_for_g = state_for_g.to(device)
        z_noise_for_g = torch.randn(BATCH_SIZE, GAN_Z_DIM, device=device)

        optimizer_g_latent.zero_grad()
        fake_latent_codes_for_g = g_latent(z_noise_for_g, state_for_g)
        g_loss = -d_latent(fake_latent_codes_for_g, state_for_g).mean()
        g_loss.backward()
        optimizer_g_latent.step()

        if (epoch + 1) % 10 == 0 or epoch == LATENT_GAN_EPOCHS -1:
            print(f"Latent GAN Epoch [{epoch+1}/{LATENT_GAN_EPOCHS}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                  f"D Real Score: {d_real_score.item():.4f}, D Fake Score: {d_fake_score.item():.4f}, GP: {gp.item():.4f}")

    print("--- Latent GAN Training Finished ---")

    # --- 3. Portfolio Optimizer Training ---
    print("\n--- Starting Portfolio Optimizer Training ---")
    g_latent.eval()
    vae.eval()

    portfolio_model = TopKOptionPortfolio(surface_shape=SURFACE_SHAPE_CONFIG, top_k=10).to(device)
    optimizer_portfolio = torch.optim.Adam(portfolio_model.parameters(), lr=PORTFOLIO_LR)

    for epoch in range(PORTFOLIO_EPOCHS):
        portfolio_model.train()

        _, returns_s, state_p, _ = generate_mock_surface( # Unpack 4
            batch_size=BATCH_SIZE, surface_shape=SURFACE_SHAPE_CONFIG
        )
        returns_s = returns_s.to(device)
        state_p = state_p.to(device)
        z_noise_p = torch.randn(BATCH_SIZE, GAN_Z_DIM, device=device)

        with torch.no_grad():
            generated_latent = g_latent(z_noise_p, state_p)
            generated_surface = vae.decode(generated_latent)

        option_weights = portfolio_model(generated_surface.detach())

        loss_s = sharpe_loss_surface(option_weights, returns_s)
        loss_c = constraint_loss(option_weights, max_weight=PORTFOLIO_MAX_WEIGHT)

        total_portfolio_loss = loss_s + PORTFOLIO_CONSTRAINT_LAMBDA * loss_c

        optimizer_portfolio.zero_grad()
        total_portfolio_loss.backward()
        optimizer_portfolio.step()

        if (epoch + 1) % 10 == 0 or epoch == PORTFOLIO_EPOCHS - 1:
            print(f"Portfolio Epoch [{epoch+1}/{PORTFOLIO_EPOCHS}], Total Loss: {total_portfolio_loss.item():.4f}, "
                  f"Sharpe Loss: {loss_s.item():.4f}, Constraint Loss: {loss_c.item():.4f}")

    print("--- Portfolio Optimizer Training Finished ---")

    print("\nAll training phases complete.")

    # --- 4. Post-Training Out-of-Sample (OOS) Evaluation ---
    print("\n--- Starting Out-of-Sample Evaluation for ATM 3M IV ---")
    vae.eval()
    g_latent.eval()
    # portfolio_model.eval() # Not directly used for IV prediction for this specific task

    oos_samples = 100
    predicted_atm_3m_ivs = []
    actual_atm_3m_ivs = []

    # Define strike and maturity axes for OOS evaluation (must match SURFACE_SHAPE_CONFIG)
    num_strikes = SURFACE_SHAPE_CONFIG[1]
    num_maturities = SURFACE_SHAPE_CONFIG[2]

    # Assuming same strike/maturity ranges as in volgan_plus_plus_v2.py for consistency
    # These need to be actual numpy arrays for get_atm_3m_iv
    oos_strikes_axis_np = np.linspace(4000, 5000, num_strikes)
    oos_maturities_axis_np = np.linspace(1/24, 1, num_maturities)

    for i in range(oos_samples):
        actual_surface, _, state, spot_raw = generate_mock_surface(
            batch_size=1,
            surface_shape=SURFACE_SHAPE_CONFIG
        )
        actual_surface = actual_surface.to(device)
        state = state.to(device)
        spot_raw_val = spot_raw.item()

        z_noise_oos = torch.randn(1, GAN_Z_DIM, device=device)

        with torch.no_grad():
            fake_latent_code = g_latent(z_noise_oos, state)
            predicted_surface = vae.decode(fake_latent_code) # predicted_surface is (1, C, S, M)

        pred_iv = get_atm_3m_iv(predicted_surface[0], spot_raw_val, oos_strikes_axis_np, oos_maturities_axis_np)
        act_iv = get_atm_3m_iv(actual_surface[0], spot_raw_val, oos_strikes_axis_np, oos_maturities_axis_np)

        if not (np.isnan(pred_iv) or np.isnan(act_iv)):
            predicted_atm_3m_ivs.append(pred_iv)
            actual_atm_3m_ivs.append(act_iv)

        if (i + 1) % 20 == 0:
            print(f"Processed OOS sample {i+1}/{oos_samples}")

    predicted_atm_3m_ivs_np = np.array(predicted_atm_3m_ivs)
    actual_atm_3m_ivs_np = np.array(actual_atm_3m_ivs)

    # Ensure matplotlib is imported for plotting
    import matplotlib.pyplot as plt

    if len(actual_atm_3m_ivs_np) > 0:
        plt.figure("OOS ATM 3M IV Prediction (VAE-GAN)", figsize=(12, 6))
        plt.plot(actual_atm_3m_ivs_np, label='Actual ATM 3M IV', marker='o', linestyle='-')
        plt.plot(predicted_atm_3m_ivs_np, label='Predicted ATM 3M IV', marker='x', linestyle='--')
        plt.title('Out-of-Sample: Actual vs. Predicted ATM 3-Month IV (VAE-GAN)')
        plt.xlabel('OOS Sample Index')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid(True)
        # plt.show() # Deferred to end

        mae = np.mean(np.abs(actual_atm_3m_ivs_np - predicted_atm_3m_ivs_np))
        rmse = np.sqrt(np.mean((actual_atm_3m_ivs_np - predicted_atm_3m_ivs_np)**2))
        mape = np.mean(np.abs((actual_atm_3m_ivs_np - predicted_atm_3m_ivs_np) / (actual_atm_3m_ivs_np + 1e-8))) * 100

        print("\nOOS ATM 3M IV Statistics (VAE-GAN):")
        print(f"  Number of valid OOS samples: {len(actual_atm_3m_ivs_np)}")
        print(f"  Mean Actual IV: {np.mean(actual_atm_3m_ivs_np):.4f}")
        print(f"  Mean Predicted IV: {np.mean(predicted_atm_3m_ivs_np):.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        if len(actual_atm_3m_ivs_np) > 1: # Need at least 2 points for correlation
            # Ensure both arrays are flat for corrcoef
            correlation_matrix = np.corrcoef(actual_atm_3m_ivs_np.flatten(), predicted_atm_3m_ivs_np.flatten())
            if correlation_matrix.ndim == 2 and correlation_matrix.shape == (2,2) : # Check if valid matrix
                correlation = correlation_matrix[0, 1]
                print(f"  Correlation: {correlation:.4f}")
                print(f"  R-squared: {correlation**2:.4f}")
            else: # Handle cases where correlation might not be calculable (e.g. constant data)
                print(f"  Correlation: NaN (Could not compute reliably)")
                print(f"  R-squared: NaN")

    else:
        print("No valid OOS IVs collected to plot or calculate stats.")

    print("\n--- End of OOS Evaluation ---")
    plt.show() # Show all plots now
