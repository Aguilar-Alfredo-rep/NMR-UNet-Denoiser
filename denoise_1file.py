# SINGLE NMR SIGNAL DENOISING (FID OR SPECTRUM)
#-------------------------------------------------------------------------------------

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hides backend info and warnings
warnings.filterwarnings("ignore")          # hides Python warnings (including some from TF)
###############################################################

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

###############################################################
from u_net import build_unet_1d
###############################################################

# Paths
###############################################################
MODEL_PATH    = glob.glob("resultados/*.h5")[0]
SCALER_X_PATH = "resultados/scaler_x.pkl"
META_PATH     = "resultados/metadata.json"
TIMES_PATH    = "times.csv"
SIG_PATH      = "1.csv"                 #  -----------------------------------> Path to the FID or FFT file to process
###############################################################

# Load metadata + scaler
#-----------------------------------------------------------------------------------------
scaler_x = joblib.load(SCALER_X_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

t_min  = float(meta["t_min"])
t_max  = float(meta["t_max"])
q_norm = float(meta["q_norm"])
#-----------------------------------------------------------------------------------------

# Time (or frequency) vector + mask
#-----------------------------------------------------------------------------------------
times = pd.read_csv(TIMES_PATH, header=None).iloc[:, 0].values.astype(float)
mask = (times >= t_min) & (times <= t_max)

# Load model
###############################################################
try:
    model = load_model(MODEL_PATH)
except Exception:
    p = int(scaler_x.mean_.shape[0])
    filtros_base  = int(meta["model"]["base_filters"])
    filtros_depth = int(meta["model"]["depth"])
    kernel        = int(meta["model"]["kernel_size"])
    drop_out      = float(meta["model"]["dropout"])
    eps           = float(meta.get("eps", 1e-8))

    model = build_unet_1d(input_length=p, n_channels=1,
                          base_filters=filtros_base, depth=filtros_depth,
                          kernel_size=kernel, dropout=drop_out)
    model.load_weights(MODEL_PATH)
###############################################################

# Signal denoising
###############################################################
#-----------------------------------------------------------------------------------------
# Extract column 1 (amplitudes) since column 0 is time/frequency
df_sig = pd.read_csv(SIG_PATH, header=None, sep=None, engine='python')
sig_raw = df_sig.iloc[:, -1].values.astype(float)

times_rec = times[mask]
sig_rec   = sig_raw[mask]

scale_sig    = max(np.quantile(sig_rec, q_norm), eps)
sig_rec_norm = sig_rec / scale_sig

sig_scaled  = scaler_x.transform(sig_rec_norm.reshape(1, -1)).astype(np.float32)

# Ejecución de inferencia y post-procesamiento (Líneas restauradas)
pred_scaled = model.predict(sig_scaled[..., None], verbose=0)[0, :, 0]
pred_norm   = scaler_x.inverse_transform(pred_scaled.reshape(1, -1)).ravel()
pred_real   = pred_norm * scale_sig
#-----------------------------------------------------------------------------------------

# Full-length reconstruction (zero-padding outside the [t_min, t_max] window)
pred_full = np.zeros_like(times, dtype=np.float64)
pred_full[mask] = pred_real

# Export CSV maintaining the original 2-column structure (X-axis, Amplitude)
#-----------------------------------------------------------------------------------------
out_data = np.column_stack((times, pred_full))
pd.DataFrame(out_data).to_csv("denoised.csv", index=False, header=False)
###############################################################

# Plot
###############################################################
plt.figure(figsize=(10,4))
plt.plot(times_rec, sig_rec, label="Noisy (raw)", linewidth=0.9, color='red')
plt.plot(times_rec, pred_real, label="Denoised (U-Net)", linewidth=1.1, color='blue')
plt.xlabel("Time (s)")    # or  Frequency (Hz)
plt.ylabel("Amplitude (a.u.)")
plt.ylim(min(np.min(sig_rec), np.min(pred_real)), max(np.max(sig_rec), np.max(pred_real)))
plt.legend()
plt.tight_layout()
plt.savefig("denoised_comparison.png", dpi=300)
plt.show()
