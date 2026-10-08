"""
Training script for the Siamese CNN.

Usage (from backend folder):
  python -m training.train_siamese --dataset_path "../dataset"

Or with full path:
  python -m training.train_siamese --dataset_path "D:/path/to/your/dataset"
"""
import argparse
import csv
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Tee:
    """Mirror stdout to a log file so every run is recorded."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

from app.config import (
    MODEL_PATH, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, MARGIN, EMBEDDING_DIM
)
from app.models.siamese_net import SiameseNetwork, ContrastiveLoss
from training.dataset import SignaturePairDataset


def train(dataset_path: str):
    # ── Logging: every run writes a full log + a per-epoch metrics CSV ──
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = Tee(log_dir / f"train_{stamp}.log")
    sys.stdout = tee
    csv_path = log_dir / f"metrics_{stamp}.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                         "val_acc", "dist_genuine", "dist_forged", "lr", "seconds"])

    try:
        _train(dataset_path, csv_writer)
    finally:
        csv_file.close()
        print(f"[Train] Log saved to:     {log_dir / f'train_{stamp}.log'}")
        print(f"[Train] Metrics CSV:      {csv_path}")
        sys.stdout = tee.terminal
        tee.close()


def _train(dataset_path: str, csv_writer):
    # ── Force CUDA if available ──
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Training will be SLOW on CPU.")
        print("   Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        response = input("Continue on CPU? (y/N): ")
        if response.lower() != "y":
            return
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # Enable CuDNN benchmark for faster training with fixed input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    print(f"[Train] Using device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Train] VRAM: {total_mem:.1f} GB")
        print(f"[Train] CuDNN benchmark: ENABLED")

    # ── Data ──
    print(f"\n[Train] Loading dataset from: {dataset_path}")
    train_data = SignaturePairDataset(dataset_path, split="train", train_ratio=0.8)
    val_data = SignaturePairDataset(dataset_path, split="val", train_ratio=0.8)

    # Use num_workers on GPU for faster data loading
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # ── Model ──
    model = SiameseNetwork(EMBEDDING_DIM).to(device)

    # ── Mixed precision training (faster on modern GPUs) ──
    # torch.amp.GradScaler exists from torch 2.4; older versions expose
    # it as torch.cuda.amp.GradScaler. Support both.
    use_amp = (device.type == "cuda")
    if use_amp:
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    criterion = ContrastiveLoss(margin=MARGIN)
    # Dual-head training: BCE on the decision head gives a direct
    # discriminative gradient that stabilizes contrastive learning.
    bce = nn.BCEWithLogitsLoss()
    # Higher weight decay: the model overfits its training writers
    # (train acc 0.97 vs val acc 0.80), so favour stronger regularization.
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    # Cosine annealing: a flat LR keeps bouncing around the minimum late
    # in training. Decaying to ~0 lets the model settle into it, which is
    # where the last few accuracy points live.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.02)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Train] Model parameters: {total_params:,} (trainable: {trainable:,})")
    print(f"[Train] Batch size: {BATCH_SIZE}")
    print(f"[Train] Learning rate: {LEARNING_RATE}")
    print(f"[Train] Epochs: {NUM_EPOCHS}")
    print(f"[Train] Contrastive margin: {MARGIN}")
    print(f"[Train] Mixed precision (AMP): {'ENABLED' if use_amp else 'disabled'}")
    print(f"[Train] Model will be saved to: {MODEL_PATH}")
    print("=" * 65)

    # ── Training Loop ──
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    emb1, emb2 = model(img1, img2)
                    logits = model.classify(emb1, emb2).squeeze(1)
                    loss = criterion(emb1, emb2, labels) + bce(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                emb1, emb2 = model(img1, img2)
                logits = model.classify(emb1, emb2).squeeze(1)
                loss = criterion(emb1, emb2, labels) + bce(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * img1.size(0)

            # Accuracy: decision-head prediction
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(".", end="", flush=True)

        train_loss /= len(train_data)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        # Separation diagnostics: mean distance per class. A healthy run
        # drives genuine down and forged up; both stuck near each other
        # means the embeddings collapsed and training is not working.
        gen_dist_sum, gen_n = 0.0, 0
        forg_dist_sum, forg_n = 0.0, 0

        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        emb1, emb2 = model(img1, img2)
                        logits = model.classify(emb1, emb2).squeeze(1)
                        loss = criterion(emb1, emb2, labels) + bce(logits, labels)
                else:
                    emb1, emb2 = model(img1, img2)
                    logits = model.classify(emb1, emb2).squeeze(1)
                    loss = criterion(emb1, emb2, labels) + bce(logits, labels)

                val_loss += loss.item() * img1.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                dist = nn.functional.pairwise_distance(emb1, emb2)
                gen_mask = labels < 0.5
                gen_dist_sum += dist[gen_mask].sum().item()
                gen_n += int(gen_mask.sum().item())
                forg_dist_sum += dist[~gen_mask].sum().item()
                forg_n += int((~gen_mask).sum().item())

        val_loss /= max(len(val_data), 1)
        val_acc = val_correct / max(val_total, 1)
        gen_dist = gen_dist_sum / max(gen_n, 1)
        forg_dist = forg_dist_sum / max(forg_n, 1)

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        csv_writer.writerow([
            epoch + 1, round(train_loss, 5), round(train_acc, 5),
            round(val_loss, 5), round(val_acc, 5), round(gen_dist, 5),
            round(forg_dist, 5), lr, round(elapsed, 1),
        ])

        print(
            f"\nEpoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"Dist gen/forg: {gen_dist:.3f}/{forg_dist:.3f} | "
            f"LR: {lr:.6f} | Time: {elapsed:.1f}s"
        )

        scheduler.step()

        # Save best model by validation ACCURACY — that is what the
        # deployed system is judged on. Val loss can improve while the
        # genuine/forged decision gets no better (and vice versa).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(MODEL_PATH))
            print(f"  * Saved best model (val_acc={val_acc:.3f}, val_loss={val_loss:.4f})")

    print("\n" + "=" * 65)
    print(f"[Train] DONE!")
    print(f"[Train] Best validation loss: {best_val_loss:.4f}")
    print(f"[Train] Best validation accuracy: {best_val_acc:.3f}")
    print(f"[Train] Model saved to: {MODEL_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    # Windows multiprocessing fix
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Train Siamese Network for Signature Verification")
    parser.add_argument(
        "--dataset_path", type=str,
        default="../dataset/CEDAR,../dataset/SYNTH,../dataset/Hindi,"
                "../dataset/archive (1)/extract",
        help='Dataset folder(s) with numbered writer subfolders. '
             'Comma-separate to combine (default: CEDAR + SYNTH + Hindi + sign_data)'
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    args = parser.parse_args()

    if args.epochs:
        NUM_EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    train(args.dataset_path)