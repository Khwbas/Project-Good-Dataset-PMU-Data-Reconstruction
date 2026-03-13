"""
=============================================================================
Autoencoder-Based PMU Data Reconstruction and Anomaly Detection
Dataset: Case III — Dynamic Frequency Event + Data Repetition Attack
Authors: Khwrwmdao Basumatary, Anup Shukla
         Indian Institute of Technology Jammu, India
=============================================================================

Architecture:
    Input (8 PMU channels, window=20)
        ↓
    Encoder: Dense(160→64→32→16) + ReLU
        ↓
    Latent Space (16-dim)
        ↓
    Decoder: Dense(16→32→64→160) + ReLU → Linear
        ↓
    Reconstructed Output (8 PMU channels, window=20)

Training:  On normal/ambient portion of data (first 3500 samples)
Testing:   On full dataset including repetition attack region
Detection: Reconstruction error (MSE) thresholded by mean + 3*std
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1.  AUTOENCODER CLASS (NumPy — no external DL framework)
# ─────────────────────────────────────────────────────────────

class DenseLayer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        # He initialisation
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.activation = activation
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == 'relu':
            self.a = np.maximum(0, self.z)
        else:                          # linear
            self.a = self.z
        return self.a

    def backward(self, dA):
        if self.activation == 'relu':
            dZ = dA * (self.z > 0)
        else:
            dZ = dA
        self.dW = self.x.T @ dZ / len(self.x)
        self.db = dZ.mean(axis=0, keepdims=True)
        return dZ @ self.W.T

    def update(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW**2
        mW_hat = self.mW / (1 - beta1**t)
        vW_hat = self.vW / (1 - beta2**t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db**2
        mb_hat = self.mb / (1 - beta1**t)
        vb_hat = self.vb / (1 - beta2**t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class Autoencoder:
    """
    Symmetric autoencoder with configurable hidden dims.
    Default: 160 → 64 → 32 → 16 → 32 → 64 → 160
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        dims = [input_dim] + hidden_dims
        # Encoder layers
        self.enc = [DenseLayer(dims[i], dims[i+1], 'relu')
                    for i in range(len(dims)-1)]
        # Decoder layers (mirror, last is linear)
        dec_dims = list(reversed(dims))
        self.dec = [DenseLayer(dec_dims[i], dec_dims[i+1],
                    'relu' if i < len(dec_dims)-2 else 'linear')
                    for i in range(len(dec_dims)-1)]
        self.layers = self.enc + self.dec

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, x, x_hat):
        # MSE loss gradient
        dA = 2 * (x_hat - x) / x.shape[1]
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def train(self, X_train, epochs=200, batch_size=64, lr=1e-3, verbose=True):
        n = len(X_train)
        losses = []
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            epoch_loss = 0
            steps = 0
            for start in range(0, n, batch_size):
                batch = X_train[idx[start:start + batch_size]]
                x_hat = self.forward(batch)
                loss = np.mean((x_hat - batch) ** 2)
                self.backward(batch, x_hat)
                t = (epoch - 1) * (n // batch_size) + steps + 1
                for layer in self.layers:
                    layer.update(lr, t)
                epoch_loss += loss
                steps += 1
            avg_loss = epoch_loss / steps
            losses.append(avg_loss)
            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"  Epoch {epoch:3d}/{epochs}  |  Train Loss: {avg_loss:.6f}")
        return losses

    def reconstruct(self, X):
        return self.forward(X)

    def reconstruction_error(self, X):
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)


# ─────────────────────────────────────────────────────────────
# 2.  DATA LOADING AND PREPARATION
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  Autoencoder PMU Data Reconstruction — Case III")
print("=" * 65)

DATA_PATH = '/mnt/user-data/uploads/Case_III_Data_Repetition_Hz__1_.csv'
WINDOW     = 20          # sliding window length (20 samples = 400 ms at 50 fps)
STEP       = 1           # stride
TRAIN_END  = 3500        # samples used for training (normal operation)
EPOCHS     = 150
BATCH_SIZE = 64
LR         = 5e-4

print("\n[1] Loading data ...")
df = pd.read_csv(DATA_PATH)
pmu_cols = [c for c in df.columns if 'PMU' in c]
# Drop repeated header rows (RTAC export quirk) and NaN rows
df_clean = df[df[pmu_cols[0]] != pmu_cols[0]].copy()
df_clean = df_clean.reset_index(drop=True)
raw = df_clean[pmu_cols].apply(pd.to_numeric, errors='coerce').dropna().values.astype(np.float32)
n_samples, n_pmu = raw.shape
print(f"    Samples: {n_samples}  |  PMU channels: {n_pmu}")
print(f"    Freq range: {raw.min():.4f} – {raw.max():.4f} Hz")

# Normalise with scaler fitted on training portion only
scaler = MinMaxScaler()
train_raw = raw[:TRAIN_END]
scaler.fit(train_raw)
data_norm = scaler.transform(raw).astype(np.float32)
# Clip to [0,1] to prevent numerical issues
data_norm = np.clip(data_norm, 0, 1)

# Build sliding windows
def make_windows(data, window, step):
    X = []
    for i in range(0, len(data) - window + 1, step):
        X.append(data[i:i + window].flatten())
    return np.array(X, dtype=np.float32)

X_all   = make_windows(data_norm, WINDOW, STEP)
X_train = make_windows(data_norm[:TRAIN_END], WINDOW, STEP)
input_dim = WINDOW * n_pmu          # 20 * 8 = 160

print(f"    Training windows : {len(X_train)}")
print(f"    Total windows    : {len(X_all)}")
print(f"    Input dimension  : {input_dim}")


# ─────────────────────────────────────────────────────────────
# 3.  BUILD AND TRAIN AUTOENCODER
# ─────────────────────────────────────────────────────────────

print("\n[2] Building Autoencoder ...")
print("    Architecture: 160 → 64 → 32 → 16 → 32 → 64 → 160")
ae = Autoencoder(input_dim=input_dim, hidden_dims=[64, 32, 16])

print(f"\n[3] Training on normal data ({len(X_train)} windows, {EPOCHS} epochs) ...")
losses = ae.train(X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
print(f"\n    Final training loss: {losses[-1]:.6f}")


# ─────────────────────────────────────────────────────────────
# 4.  RECONSTRUCTION AND ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────

print("\n[4] Reconstructing full dataset ...")
X_hat_norm = ae.reconstruct(X_all)

# Convert reconstructed windows back to per-sample (average overlapping windows)
recon_norm = np.zeros_like(data_norm)
counts     = np.zeros(n_samples)
for i, start in enumerate(range(0, n_samples - WINDOW + 1, STEP)):
    recon_norm[start:start + WINDOW] += X_hat_norm[i].reshape(WINDOW, n_pmu)
    counts[start:start + WINDOW]     += 1
counts = np.maximum(counts, 1)
recon_norm /= counts[:, None]

# Inverse transform to Hz
recon_hz = scaler.inverse_transform(np.clip(recon_norm, 0, 1).astype(np.float32))

# Per-window reconstruction error
recon_err = ae.reconstruction_error(X_all)

# Anomaly threshold: mean + 3*std on training portion
train_err   = ae.reconstruction_error(X_train)
threshold   = train_err.mean() + 3 * train_err.std()
anomalies   = recon_err > threshold

# Map window index back to sample index (centre of window)
win_to_sample = np.array([i + WINDOW // 2 for i in range(len(X_all))])

print(f"    Anomaly threshold   : {threshold:.6f}")
print(f"    Anomalous windows   : {anomalies.sum()} / {len(anomalies)}"
      f"  ({100*anomalies.mean():.1f}%)")

# Compute per-channel RMSE
rmse_per_channel = np.sqrt(np.mean((raw - recon_hz)**2, axis=0))
print(f"\n    Per-channel RMSE (Hz):")
for i, (col, rmse) in enumerate(zip(pmu_cols, rmse_per_channel)):
    print(f"      {col[:10]:12s}: {rmse:.5f} Hz")
print(f"    Overall RMSE: {np.sqrt(np.mean((raw - recon_hz)**2)):.5f} Hz")


# ─────────────────────────────────────────────────────────────
# 5.  PLOT RESULTS
# ─────────────────────────────────────────────────────────────

print("\n[5] Generating plots ...")

t = np.arange(n_samples) / 50.0   # time axis in seconds
PURPLE = '#79064B'
BLUE   = '#1f77b4'
RED    = '#d62728'
GREEN  = '#2ca02c'
ORANGE = '#ff7f0e'

fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.3)

# ── (a) Raw vs Reconstructed — PMU01 ──────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, raw[:, 0], color=BLUE, lw=0.8, label='Original PMU01', alpha=0.9)
ax1.plot(t, recon_hz[:, 0], color=RED, lw=1.0, ls='--',
         label='Reconstructed PMU01', alpha=0.9)
ax1.axvline(TRAIN_END/50, color='grey', ls=':', lw=1.2, label='Train/Test boundary')
ax1.axvspan(TRAIN_END/50, n_samples/50, alpha=0.06, color=ORANGE, label='Attack region')
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Frequency (Hz)', fontsize=10)
ax1.set_title('(a) Original vs Autoencoder Reconstructed Signal — PMU01 (Case III: Data Repetition Attack)',
              fontsize=11, fontweight='bold', color=PURPLE)
ax1.legend(fontsize=8, loc='lower left')
ax1.grid(True, alpha=0.3)

# ── (b) All 8 PMU channels reconstructed ─────────────────────
colors8 = plt.cm.tab10(np.linspace(0, 0.8, n_pmu))
ax2 = fig.add_subplot(gs[1, :])
for i in range(n_pmu):
    ax2.plot(t, recon_hz[:, i], lw=0.7, color=colors8[i],
             label=f'PMU{i+1:02d}', alpha=0.85)
ax2.axvline(TRAIN_END/50, color='grey', ls=':', lw=1.2)
ax2.axvspan(TRAIN_END/50, n_samples/50, alpha=0.06, color=ORANGE)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Frequency (Hz)', fontsize=10)
ax2.set_title('(b) Reconstructed Frequency — All 8 PMU Channels', fontsize=11,
              fontweight='bold', color=PURPLE)
ax2.legend(fontsize=7, ncol=4, loc='lower left')
ax2.grid(True, alpha=0.3)

# ── (c) Reconstruction error + threshold ──────────────────────
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(win_to_sample / 50, recon_err, color=BLUE, lw=0.7,
         label='Reconstruction Error (MSE)', alpha=0.85)
ax3.axhline(threshold, color=RED, lw=1.5, ls='--',
            label=f'Threshold = {threshold:.5f}')
ax3.fill_between(win_to_sample / 50, recon_err, threshold,
                 where=anomalies, color=RED, alpha=0.3, label='Detected anomaly')
ax3.axvline(TRAIN_END/50, color='grey', ls=':', lw=1.2, label='Train/Test boundary')
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('MSE', fontsize=10)
ax3.set_title('(c) Reconstruction Error and Anomaly Detection',
              fontsize=11, fontweight='bold', color=PURPLE)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── (d) Training loss curve ────────────────────────────────────
ax4 = fig.add_subplot(gs[3, 0])
ax4.plot(range(1, EPOCHS+1), losses, color=PURPLE, lw=1.5)
ax4.set_xlabel('Epoch', fontsize=10)
ax4.set_ylabel('MSE Loss', fontsize=10)
ax4.set_title('(d) Autoencoder Training Loss', fontsize=11,
              fontweight='bold', color=PURPLE)
ax4.grid(True, alpha=0.3)

# ── (e) Per-channel RMSE bar chart ────────────────────────────
ax5 = fig.add_subplot(gs[3, 1])
ch_labels = [f'PMU{i+1:02d}' for i in range(n_pmu)]
bars = ax5.bar(ch_labels, rmse_per_channel, color=PURPLE, alpha=0.8, edgecolor='white')
ax5.set_xlabel('PMU Channel', fontsize=10)
ax5.set_ylabel('RMSE (Hz)', fontsize=10)
ax5.set_title('(e) Per-Channel Reconstruction RMSE', fontsize=11,
              fontweight='bold', color=PURPLE)
for bar, val in zip(bars, rmse_per_channel):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7)
ax5.grid(True, alpha=0.3, axis='y')

# ── (f) Residual (original − reconstructed) PMU01 ────────────
ax6 = fig.add_subplot(gs[4, :])
residual = raw[:, 0] - recon_hz[:, 0]
ax6.plot(t, residual, color=GREEN, lw=0.7, label='Residual (Original − Reconstructed)', alpha=0.9)
ax6.axhline(0, color='black', lw=0.8, ls='--')
ax6.axvline(TRAIN_END/50, color='grey', ls=':', lw=1.2, label='Train/Test boundary')
ax6.axvspan(TRAIN_END/50, n_samples/50, alpha=0.06, color=ORANGE, label='Attack region')
ax6.set_xlabel('Time (s)', fontsize=10)
ax6.set_ylabel('Residual (Hz)', fontsize=10)
ax6.set_title('(f) Residual Signal — PMU01 (Highlighting Attack Region)',
              fontsize=11, fontweight='bold', color=PURPLE)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

fig.suptitle('Autoencoder-Based PMU Data Reconstruction and Anomaly Detection\n'
             'Case III: Dynamic Frequency Event + Data Repetition Attack  |  IIT Jammu',
             fontsize=13, fontweight='bold', color=PURPLE, y=0.98)

plt.savefig('/home/claude/autoencoder_results.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("    Saved: autoencoder_results.png")


# ─────────────────────────────────────────────────────────────
# 6.  SAVE RECONSTRUCTED DATA
# ─────────────────────────────────────────────────────────────

print("\n[6] Saving reconstructed dataset ...")
out_df = df_clean[df_clean[pmu_cols[0]].apply(lambda x: str(x) != pmu_cols[0])].copy()
out_df = out_df.reset_index(drop=True)
short_cols = [f'PMU{i+1:02d}' for i in range(n_pmu)]
for i, col in enumerate(short_cols):
    out_df[f'{col}_reconstructed'] = np.round(recon_hz[:, i], 6)
    out_df[f'{col}_residual']      = np.round(raw[:, i] - recon_hz[:, i], 6)

# Add anomaly flag per sample
sample_anomaly = np.zeros(n_samples, dtype=int)
for i, start in enumerate(range(0, n_samples - WINDOW + 1, STEP)):
    if anomalies[i]:
        sample_anomaly[start:start + WINDOW] = 1
out_df['anomaly_flag'] = sample_anomaly

out_df.to_csv('/home/claude/Case_III_Reconstructed.csv', index=False)
print("    Saved: Case_III_Reconstructed.csv")

print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  Model           : Autoencoder (NumPy, Adam optimiser)")
print(f"  Architecture    : 160 → 64 → 32 → 16 → 32 → 64 → 160")
print(f"  Window size     : {WINDOW} samples ({WINDOW*20} ms)")
print(f"  Training epochs : {EPOCHS}")
print(f"  Training loss   : {losses[-1]:.6f}")
print(f"  Overall RMSE    : {np.sqrt(np.mean((raw - recon_hz)**2)):.5f} Hz")
print(f"  Anomaly thresh  : {threshold:.6f}")
print(f"  Anomalies found : {anomalies.sum()} windows ({100*anomalies.mean():.1f}%)")
print("=" * 65)
