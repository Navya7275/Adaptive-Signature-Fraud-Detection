"""
Configuration — all settings, paths, and thresholds in one place.
"""
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNATURES_DIR = DATA_DIR / "signatures"
MODELS_DIR = BASE_DIR / "trained_models"
DB_PATH = DATA_DIR / "signatures.db"

# Create dirs on import
for d in [SIGNATURES_DIR, MODELS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Image Processing ──
IMG_HEIGHT = 150
IMG_WIDTH = 220
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
EMBEDDING_DIM = 128

# ── Siamese Network Training ──
MODEL_PATH = MODELS_DIR / "siamese_model.pth"
LEARNING_RATE = 0.0004
BATCH_SIZE = 32
NUM_EPOCHS = 20
# Contrastive loss margin — applied to UNNORMALIZED embeddings (the
# SigNet recipe). Do not L2-normalize during training: on a unit sphere
# the margin becomes geometrically infeasible for many writers and the
# model settles into a constant-distance blob at loss = 0.5*(margin/2)^2.
MARGIN = 1.0

# ── Verification Thresholds ──
# Calibrated from training/evaluate.py on held-out writers (writer-disjoint).
# Final model: genuine pairs land at cosine ~0.94, skilled forgeries ~0.80
# (ROC-AUC 0.884, EER 18.4% combined; 0.944 / 10.1% on CEDAR+SYNTH+Hindi).
# Re-run `python -m training.evaluate` after every retrain and update this.
BASE_THRESHOLD = 0.90
MAX_TRUST_BUFFER = 0.03
TENURE_BUFFER_RATE = 0.0015    # per month
ESCALATION_BAND = 0.05

# ── Drift Analysis ──
MIN_SAMPLES_FOR_DRIFT = 5
DRIFT_WINDOW_MONTHS = 24
AGING_DRIFT_MAX_SLOPE = -0.003
FORGERY_IMPROVEMENT_SLOPE = 0.02
# Genuine scores sit near 0.94 with roughly 0.03 spread, so a sustained
# drop of 0.10 is a major change and volatility past 0.06 is unusual.
MEDICAL_EVENT_VOLATILITY = 0.06
MEDICAL_EVENT_DROP = 0.10
SHORT_WINDOW = 5
LONG_WINDOW = 20

# ── Stroke Features ──
TREMOR_FREQ_THRESHOLD = 8.0    # Hz — above this = pathological tremor
PRESSURE_BINS = 10