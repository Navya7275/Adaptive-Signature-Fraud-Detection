"""
Generate the project's results: metrics tables + publication-style figures.

METHODOLOGY (important):
The model is trained on a combined corpus with one writer-disjoint 80/20
split. This script evaluates ONLY on that same held-out 20% and then
groups the results by source dataset. It deliberately does NOT re-split
an individual dataset, because re-splitting puts writers the model
trained on into that dataset's "validation" set and inflates the score
(measured: AUC 0.98+ when contaminated vs the honest value below).

Writes to ../results/:
  roc_curves.png          ROC per source dataset, held-out writers only
  score_distributions.png genuine vs forged similarity histograms
  training_curves.png     train/val accuracy + loss per epoch
  metrics.json            machine-readable results
  METRICS.md              markdown tables for the README

Usage (from backend folder):
  python -m training.generate_results
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODEL_PATH, EMBEDDING_DIM, BATCH_SIZE, BASE_THRESHOLD
from app.models.siamese_net import SiameseNetwork
from training.dataset import SignaturePairDataset

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# The exact corpus the model was trained on, in order. SignaturePairDataset
# offsets writer IDs by 1000 per dataset, so the offset identifies the source.
COMBINED = ("../dataset/CEDAR,../dataset/SYNTH,../dataset/Hindi,"
            "../dataset/archive (1)/extract")
SOURCES = {
    0: "CEDAR (English)",
    1: "SYNTH (generated)",
    2: "BHSig260 (Hindi)",
    3: "sign_data (real)",
}
COLORS = {
    "All held-out writers": "#4f46e5",
    "CEDAR (English)": "#059669",
    "SYNTH (generated)": "#7c3aed",
    "BHSig260 (Hindi)": "#d97706",
    "sign_data (real)": "#dc2626",
}


def cosine_from_distance(d):
    """Embeddings are L2-normalized: D^2 = 2 - 2cos  =>  cos = 1 - D^2/2."""
    return 1.0 - (d ** 2) / 2.0


def distance_from_cosine(c):
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - c))))


def compute_metrics(dists, labels, n_writers):
    """labels: 0 = genuine, 1 = forged. Larger distance => more likely forged."""
    gen_mask = labels < 0.5
    auc = roc_auc_score(labels, dists)
    fpr, tpr, thr = roc_curve(labels, dists)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2)

    accs = [((dists > t) == (labels >= 0.5)).mean() for t in thr]
    best_i = int(np.argmax(accs))

    # Operating point actually deployed in the app (config BASE_THRESHOLD)
    op = distance_from_cosine(BASE_THRESHOLD)
    return {
        "writers": int(n_writers),
        "pairs": int(len(dists)),
        "genuine_pairs": int(gen_mask.sum()),
        "forged_pairs": int((~gen_mask).sum()),
        "roc_auc": float(auc),
        "eer": eer,
        "best_accuracy": float(accs[best_i]),
        "best_threshold_cosine": float(cosine_from_distance(thr[best_i])),
        "genuine_similarity_mean": float(cosine_from_distance(dists[gen_mask]).mean()),
        "forged_similarity_mean": float(cosine_from_distance(dists[~gen_mask]).mean()),
        "far_at_deployed": float((dists[~gen_mask] <= op).mean()),
        "frr_at_deployed": float((dists[gen_mask] > op).mean()),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def plot_roc(results):
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        main = name.startswith("All")
        plt.plot(r["roc"]["fpr"], r["roc"]["tpr"],
                 label=f'{name} — AUC {r["roc_auc"]:.3f}',
                 color=COLORS[name], linewidth=2.6 if main else 1.7,
                 alpha=1.0 if main else 0.85, zorder=3 if main else 2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4,
             label="Random (AUC 0.500)")
    plt.xlabel("False Accept Rate", fontsize=11)
    plt.ylabel("True Reject Rate", fontsize=11)
    plt.title("Forgery Detection ROC — Held-Out Writers Only",
              fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=160)
    plt.close()


def plot_distributions(dists, labels):
    gen = cosine_from_distance(dists[labels < 0.5])
    forg = cosine_from_distance(dists[labels >= 0.5])
    plt.figure(figsize=(8, 5))
    bins = np.linspace(min(forg.min(), gen.min()), 1.0, 60)
    plt.hist(gen, bins=bins, alpha=0.72, label=f"Genuine pairs (n={len(gen):,})",
             color="#059669", edgecolor="white", linewidth=0.3)
    plt.hist(forg, bins=bins, alpha=0.72, label=f"Forged pairs (n={len(forg):,})",
             color="#dc2626", edgecolor="white", linewidth=0.3)
    plt.axvline(BASE_THRESHOLD, color="#1f2937", linestyle="--", linewidth=2,
                label=f"Deployed threshold ({BASE_THRESHOLD})")
    plt.xlabel("Cosine similarity to reference", fontsize=11)
    plt.ylabel("Number of pairs", fontsize=11)
    plt.title("Score Separation — Genuine vs Forged (held-out writers)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "score_distributions.png", dpi=160)
    plt.close()


def plot_training_curves():
    logs = sorted((Path(__file__).resolve().parent.parent / "logs").glob("metrics_*.csv"))
    if not logs:
        print("[Results] No metrics CSV found — skipping training curves")
        return
    rows = list(csv.DictReader(open(logs[-1], encoding="utf-8")))
    if len(rows) < 2:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(ep, [float(r["train_acc"]) for r in rows], label="Train",
            color="#4f46e5", linewidth=2)
    a1.plot(ep, [float(r["val_acc"]) for r in rows], label="Validation",
            color="#dc2626", linewidth=2)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy")
    a1.set_title("Accuracy", fontweight="bold"); a1.legend(); a1.grid(alpha=0.25)
    a2.plot(ep, [float(r["train_loss"]) for r in rows], label="Train",
            color="#4f46e5", linewidth=2)
    a2.plot(ep, [float(r["val_loss"]) for r in rows], label="Validation",
            color="#dc2626", linewidth=2)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Loss")
    a2.set_title("Loss", fontweight="bold"); a2.legend(); a2.grid(alpha=0.25)
    fig.suptitle("Training history (20 epochs, cosine LR schedule)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


def write_markdown(results):
    L = [
        "# Results",
        "",
        "All numbers below are measured on **held-out writers only** — the",
        "writer-disjoint 20% validation split of the training corpus. No writer",
        "in these tables appears anywhere in training, so this reflects",
        "generalization to people the model has never seen.",
        "",
        "Per-dataset rows are *subsets of that same split*, not fresh splits of",
        "each dataset. (Re-splitting an individual dataset leaks training writers",
        "into its validation set and inflates AUC to 0.98+ — avoided here.)",
        "",
        "## Verification performance",
        "",
        "| Evaluation set | Writers | Pairs | ROC-AUC | EER | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        L.append(f'| {name} | {r["writers"]} | {r["pairs"]:,} | '
                 f'{r["roc_auc"]:.3f} | {r["eer"]*100:.1f}% | '
                 f'{r["best_accuracy"]*100:.1f}% |')

    L += ["", "## Score separation", "",
          "| Evaluation set | Genuine similarity | Forged similarity | Gap |",
          "|---|---:|---:|---:|"]
    for name, r in results.items():
        g, f = r["genuine_similarity_mean"], r["forged_similarity_mean"]
        L.append(f"| {name} | {g:.3f} | {f:.3f} | **{g-f:.3f}** |")

    L += ["", f"## Operating point (deployed threshold = {BASE_THRESHOLD} cosine)",
          "",
          "| Evaluation set | False Accept Rate | False Reject Rate |",
          "|---|---:|---:|"]
    for name, r in results.items():
        L.append(f'| {name} | {r["far_at_deployed"]*100:.1f}% | '
                 f'{r["frr_at_deployed"]*100:.1f}% |')

    L += [
        "",
        "Borderline scores inside the escalation band are routed to human review",
        "rather than auto-rejected, so the false-reject figures above are an upper",
        "bound on user-visible rejections.",
        "",
        "## Figures",
        "",
        "![ROC curves](roc_curves.png)",
        "",
        "![Score distributions](score_distributions.png)",
        "",
        "![Training curves](training_curves.png)",
        "",
    ]
    (RESULTS_DIR / "METRICS.md").write_text("\n".join(L), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(EMBEDDING_DIM)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device,
                                     weights_only=True))
    model = model.to(device).eval()
    print(f"[Results] Model: {MODEL_PATH.name} on {device}\n")

    data = SignaturePairDataset(COMBINED, split="val", train_ratio=0.8)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    dists, labels = [], []
    with torch.no_grad():
        for img1, img2, lbl in loader:
            e1, e2 = model(img1.to(device), img2.to(device))
            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
            dists.append(torch.nn.functional.pairwise_distance(e1, e2).cpu().numpy())
            labels.append(lbl.numpy())
    dists = np.concatenate(dists)
    labels = np.concatenate(labels)
    writers = np.array(data.pair_writers)
    assert len(writers) == len(dists), "pair/writer mismatch"

    results = {"All held-out writers": compute_metrics(
        dists, labels, len(set(writers)))}
    r = results["All held-out writers"]
    print(f'[Results] All held-out  : AUC {r["roc_auc"]:.3f} | '
          f'EER {r["eer"]*100:.1f}% | acc {r["best_accuracy"]*100:.1f}% '
          f'({r["writers"]} writers)')

    for offset, name in SOURCES.items():
        mask = (writers // 1000) == offset
        if mask.sum() == 0:
            continue
        results[name] = compute_metrics(
            dists[mask], labels[mask], len(set(writers[mask])))
        r = results[name]
        print(f'[Results] {name:18s}: AUC {r["roc_auc"]:.3f} | '
              f'EER {r["eer"]*100:.1f}% | acc {r["best_accuracy"]*100:.1f}% '
              f'({r["writers"]} writers)')

    plot_roc(results)
    plot_distributions(dists, labels)
    plot_training_curves()

    slim = {k: {kk: vv for kk, vv in v.items() if kk != "roc"}
            for k, v in results.items()}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(slim, indent=2),
                                              encoding="utf-8")
    write_markdown(results)

    print(f"\n[Results] Written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
