"""
Model evaluation — the metrics that matter for verification.

The training log's accuracy uses a fixed distance threshold (MARGIN/2),
which badly understates a model whose distances live elsewhere. This
script loads the saved best model and computes, on the held-out
writer-disjoint validation split:

  - ROC-AUC          (threshold-independent separability)
  - EER              (equal error rate — the standard biometric metric)
  - Best threshold   (distance + accuracy at that threshold)
  - Suggested BASE_THRESHOLD for app/config.py (cosine-similarity space)

Usage (from backend folder):
  python -m training.evaluate
  python -m training.evaluate --dataset_path "../dataset/CEDAR"   # single set
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODEL_PATH, EMBEDDING_DIM, BATCH_SIZE
from app.models.siamese_net import SiameseNetwork
from training.dataset import SignaturePairDataset


def evaluate(dataset_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    if not MODEL_PATH.exists():
        print(f"[Eval] ERROR: no model at {MODEL_PATH} — train first")
        return

    model = SiameseNetwork(EMBEDDING_DIM)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device,
                                     weights_only=True))
    model = model.to(device).eval()
    print(f"[Eval] Loaded {MODEL_PATH}")

    val_data = SignaturePairDataset(dataset_path, split="val", train_ratio=0.8)
    loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    dists, labels, probs = [], [], []
    with torch.no_grad():
        for img1, img2, lbl in loader:
            emb1, emb2 = model(img1.to(device), img2.to(device))
            # Decision-head probability of 'forged'
            logit = model.classify(emb1, emb2).squeeze(1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            # Distance in the app's space: normalized embeddings (cosine)
            e1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
            d = torch.nn.functional.pairwise_distance(e1, e2)
            dists.append(d.cpu().numpy())
            labels.append(lbl.numpy())

    dists = np.concatenate(dists)
    labels = np.concatenate(labels)          # 0 = genuine, 1 = forged
    probs = np.concatenate(probs)
    gen, forg = dists[labels < 0.5], dists[labels >= 0.5]

    print(f"\n[Eval] Pairs: {len(dists)} ({len(gen)} genuine, {len(forg)} forged)")
    print(f"[Eval] Genuine distance: mean {gen.mean():.4f}  std {gen.std():.4f}")
    print(f"[Eval] Forged  distance: mean {forg.mean():.4f}  std {forg.std():.4f}")

    # ── ROC-AUC ──
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(labels, dists)       # higher dist should mean forged
    print(f"\n[Eval] ROC-AUC: {auc:.4f}   (1.0 = perfect, 0.5 = random)")

    # ── EER + best threshold ──
    fpr, tpr, thresholds = roc_curve(labels, dists)
    fnr = 1 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"[Eval] EER: {eer * 100:.2f}%   (at distance threshold "
          f"{thresholds[eer_idx]:.4f})")

    accs = [((dists > t) == (labels >= 0.5)).mean() for t in thresholds]
    best_i = int(np.argmax(accs))
    best_t = float(thresholds[best_i])
    print(f"[Eval] Best accuracy: {accs[best_i] * 100:.2f}% at distance "
          f"threshold {best_t:.4f}")

    # ── Decision head (classifier) metrics ──
    cls_auc = roc_auc_score(labels, probs)
    cls_acc = max(((probs > t) == (labels >= 0.5)).mean()
                  for t in np.arange(0.05, 1.0, 0.05))
    print(f"\n[Eval] Decision head ROC-AUC: {cls_auc:.4f}   "
          f"best accuracy: {cls_acc * 100:.2f}%")

    # ── Suggested config value ──
    # The app decides on cosine similarity of L2-normalized embeddings:
    # D^2 = 2 - 2*cos  =>  cos = 1 - D^2 / 2
    cos_thr = 1 - best_t ** 2 / 2
    print(f"\n[Eval] Suggested BASE_THRESHOLD for app/config.py: {cos_thr:.2f}")
    print("[Eval] (per-user thresholds calibrate around this at enrollment)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained Siamese model")
    parser.add_argument(
        "--dataset_path", type=str,
        default="../dataset/CEDAR,../dataset/SYNTH,../dataset/Hindi,"
                "../dataset/archive (1)/extract",
    )
    args = parser.parse_args()
    evaluate(args.dataset_path)
