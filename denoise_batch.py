# FULL BATCH DENOISING OF NMR SIGNALS OR SPECTRA
#-----------------------------------------------------------------------------------------

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hides backend INFO and WARNING messages
warnings.filterwarnings("ignore")          # hides Python warnings (including some from TF)
###############################################################

import os, glob, json
import numpy as np
import pandas as pd
from tqdm import tqdm
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

IN_DIR  = "FIDs"   # or FFTs               #  -------------------------------------------------------------------> Input folder
OUT_DIR = "files_denoised"           # Output folder
os.makedirs(OUT_DIR, exist_ok=True)
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

# Time vector + mask
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

# Batch execution
###############################################################
files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))

for fpath in tqdm(files, desc="Denoising ...", ncols=100, leave=True, dynamic_ncols=True):

    # Read amplitude (Auto-detect delimiter, extract last column)
    df_sig = pd.read_csv(fpath, header=None, sep=None, engine='python')
    sig_raw = df_sig.iloc[:, -1].values.astype(float)

    sig_rec = sig_raw[mask]

    # Preprocessing
    scale_sig = max(np.quantile(sig_rec, q_norm), eps)
    sig_rec_norm = sig_rec / scale_sig
    sig_scaled = scaler_x.transform(sig_rec_norm.reshape(1, -1)).astype(np.float32)

    # Inference
    pred_scaled = model.predict(sig_scaled[..., None], verbose=0)[0, :, 0]

    # Postprocessing
    pred_norm = scaler_x.inverse_transform(pred_scaled.reshape(1, -1)).ravel()
    pred_real = pred_norm * scale_sig

    # Full-length array reconstruction (zero-padding outside the time mask)
    pred_full = np.zeros_like(times, dtype=np.float64)
    pred_full[mask] = pred_real

    # Matrix writing [Time, Amplitude]
    out_path = os.path.join(OUT_DIR, os.path.basename(fpath))
    np.savetxt(out_path, np.column_stack((times, pred_full)), delimiter=",", fmt="%.8e")

###############################################################
print("\nReconstruction pipeline finished successfully.")
