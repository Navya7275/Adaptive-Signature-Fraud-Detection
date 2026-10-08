import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import numpy as np
from app.models.embedding import load_model, extract_embedding

def test_dataset(name, dataset_path, writer_range, genuine_glob, forged_glob):
    print(f"\n{'='*60}")
    print(f"  TESTING: {name}")
    print(f"{'='*60}")

    genuine_scores = []
    forged_scores = []

    for wid in writer_range:
        wdir = dataset_path / str(wid)
        if not wdir.exists():
            continue

        if callable(genuine_glob):
            genuines = genuine_glob(wdir)
            forgeries = forged_glob(wdir)
        else:
            genuines = sorted(wdir.glob(genuine_glob))
            forgeries = sorted(wdir.glob(forged_glob))

        if len(genuines) < 5 or len(forgeries) < 5:
            continue

        for i in range(min(5, len(genuines)-5)):
            e1 = extract_embedding(str(genuines[i]))
            e2 = extract_embedding(str(genuines[i+5]))
            t1 = torch.from_numpy(e1.copy()).unsqueeze(0).float()
            t2 = torch.from_numpy(e2.copy()).unsqueeze(0).float()
            sim = F.cosine_similarity(t1, t2).item()
            genuine_scores.append(sim)

        for i in range(min(5, len(forgeries))):
            e1 = extract_embedding(str(genuines[i]))
            e2 = extract_embedding(str(forgeries[i]))
            t1 = torch.from_numpy(e1.copy()).unsqueeze(0).float()
            t2 = torch.from_numpy(e2.copy()).unsqueeze(0).float()
            sim = F.cosine_similarity(t1, t2).item()
            forged_scores.append(sim)

    if not genuine_scores or not forged_scores:
        print("  No test data found!")
        return

    g = np.array(genuine_scores)
    f = np.array(forged_scores)

    print(f"\n  Genuine pairs ({len(g)}): mean={g.mean():.3f}, std={g.std():.3f}")
    print(f"  Forged pairs  ({len(f)}): mean={f.mean():.3f}, std={f.std():.3f}")
    print(f"  Gap: {g.mean() - f.mean():.3f}")

    best_acc, best_t = 0, 0
    for t in np.arange(0.3, 0.99, 0.01):
        correct = (g >= t).sum() + (f < t).sum()
        acc = correct / (len(g) + len(f))
        if acc > best_acc:
            best_acc, best_t = acc, t

    print(f"  BEST ACCURACY: {best_acc*100:.1f}% at threshold {best_t:.2f}")
    return best_acc

def cedar_genuine(wdir):
    return sorted(wdir.glob("original_*.png"))

def cedar_forged(wdir):
    return sorted(wdir.glob("forgeries_*.png"))

def hindi_genuine(wdir):
    return sorted([f for f in wdir.iterdir() if "-G-" in f.name.upper()])

def hindi_forged(wdir):
    return sorted([f for f in wdir.iterdir() if "-F-" in f.name.upper()])

if __name__ == "__main__":
    model = load_model()
    print(f"Device: {next(model.parameters()).device}")

    test_dataset("CEDAR (held-out writers 45-55)",
                 Path("../dataset/CEDAR"), range(45, 56),
                 cedar_genuine, cedar_forged)

    test_dataset("Hindi (held-out writers 145-160)",
                 Path("../dataset/Hindi"), range(145, 161),
                 hindi_genuine, hindi_forged)