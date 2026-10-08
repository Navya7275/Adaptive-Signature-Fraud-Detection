import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.signature_proc import preprocess_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def augment_signature(tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """
    Random augmentation for training robustness (phone photos, pen/paper
    variation). Input/output: tensor (1, H, W), values in [0, 1],
    INVERTED polarity (ink = 1, background = 0 — see preprocess_image).
    """
    img = tensor.squeeze(0).numpy()
    h, w = img.shape

    # Small rotation + scale (capture angle differences)
    angle = rng.uniform(-4, 4)
    scale = rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)

    # Elastic wobble (hand/pen variation)
    if rng.random() < 0.5:
        alpha = rng.uniform(4, 10)
        dx = cv2.GaussianBlur(
            np.random.rand(h, w).astype(np.float32) * 2 - 1, (0, 0), 6) * alpha
        dy = cv2.GaussianBlur(
            np.random.rand(h, w).astype(np.float32) * 2 - 1, (0, 0), 6) * alpha
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        img = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0.0)

    # Stroke thickness (pen tip / pressure) — ink is bright now,
    # so dilate thickens strokes and erode thins them
    r = rng.random()
    if r < 0.25:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    elif r < 0.5:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))

    # ── Phone-photo simulation ──
    # Perspective tilt (camera not perfectly overhead)
    if rng.random() < 0.4:
        jit = 0.04
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32(
            [[rng.uniform(-jit, jit) * w, rng.uniform(-jit, jit) * h]
             for _ in range(4)])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=0.0)

    # Uneven illumination / shadows (low-frequency additive field —
    # in ink-high polarity a shadow lifts the background above zero)
    if rng.random() < 0.4:
        field = np.random.rand(6, 8).astype(np.float32)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img + field * rng.uniform(0.05, 0.18)

    # Ink intensity (lighting, ink color in grayscale)
    img = img * rng.uniform(0.6, 1.0)

    # Sensor noise
    if rng.random() < 0.5:
        img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)

    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # JPEG compression artifacts (phone cameras / messaging apps)
    if rng.random() < 0.4:
        q = int(rng.uniform(30, 80))
        ok, enc = cv2.imencode(
            ".jpg", (img * 255).astype(np.uint8),
            [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0)


class SignaturePairDataset(Dataset):
    def __init__(self, dataset_root, split="train", train_ratio=0.8):
        """
        dataset_root can be:
          - A single folder (e.g., ../dataset/Hindi)
          - A comma-separated list (e.g., ../dataset/CEDAR,../dataset/Hindi)
        """
        self.genuine_by_writer = defaultdict(list)
        self.forged_by_writer = defaultdict(list)
        self.augment = (split == "train")
        self._rng = random.Random()

        # Support multiple datasets separated by comma
        paths = [Path(p.strip()) for p in str(dataset_root).split(",")]
        writer_offset = 0

        for dataset_path in paths:
            if not dataset_path.exists():
                print(f"[Dataset] WARNING: {dataset_path} not found, skipping")
                continue

            writer_dirs = sorted(
                [d for d in dataset_path.iterdir() if d.is_dir()],
                key=lambda x: int(x.name) if x.name.isdigit() else 0
            )

            if len(writer_dirs) == 0:
                continue

            # Auto-detect format
            first_files = list(writer_dirs[0].iterdir())
            first_names = [f.name.lower() for f in first_files[:5]]
            is_bhsig = any("h-s-" in n for n in first_names)
            # "sign_data" style: genuine in NNN/, forgeries in NNN_forg/
            is_forgdir = any(d.name.endswith("_forg") for d in writer_dirs[:50]) or \
                any((dataset_path / f"{d.name}_forg").exists() for d in writer_dirs[:5])

            fmt = "BHSig260" if is_bhsig else ("forg-dirs" if is_forgdir else "CEDAR")
            print(f"[Dataset] Loading {dataset_path} ({fmt}, {len(writer_dirs)} writers, offset={writer_offset})")

            for writer_dir in writer_dirs:
                if is_forgdir:
                    # Folder name defines writer + label; filenames vary
                    name = writer_dir.name
                    if name.endswith("_forg"):
                        base = name[:-5]
                        if not base.isdigit():
                            continue
                        wid = int(base) + writer_offset
                        target = self.forged_by_writer[wid]
                    elif name.isdigit():
                        wid = int(name) + writer_offset
                        target = self.genuine_by_writer[wid]
                    else:
                        continue
                    for f in sorted(writer_dir.iterdir()):
                        # skip desktop.ini / Thumbs.db and other non-images
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                            target.append(str(f))
                    continue

                if not writer_dir.name.isdigit():
                    continue
                # Offset writer IDs so CEDAR writer 1 != Hindi writer 1
                wid = int(writer_dir.name) + writer_offset

                for f in sorted(writer_dir.iterdir()):
                    if not f.is_file():
                        continue

                    if is_bhsig:
                        upper = f.name.upper()
                        if "-G-" in upper:
                            self.genuine_by_writer[wid].append(str(f))
                        elif "-F-" in upper:
                            self.forged_by_writer[wid].append(str(f))
                    else:
                        lower = f.stem.lower()
                        if lower.startswith("original"):
                            self.genuine_by_writer[wid].append(str(f))
                        elif lower.startswith("forgeries") or lower.startswith("forged"):
                            self.forged_by_writer[wid].append(str(f))

            writer_offset += 1000  # large offset to prevent ID collision

        all_writers = sorted(
            set(self.genuine_by_writer.keys()) & set(self.forged_by_writer.keys())
        )
        # Shuffle with a FIXED seed before the train/val split so every
        # dataset (CEDAR, SYNTH, Hindi, ...) is represented in both splits.
        # Without this, writers sort by ID and the val split becomes
        # whichever dataset got the highest offset. Fixed seed keeps
        # train/val writer-disjoint across runs.
        random.Random(1234).shuffle(all_writers)

        print(f"[Dataset] Total writers with genuine: {len(self.genuine_by_writer)}")
        print(f"[Dataset] Total writers with forged: {len(self.forged_by_writer)}")
        print(f"[Dataset] Total writers with BOTH: {len(all_writers)}")

        if len(all_writers) == 0:
            raise ValueError("No writers found with both genuine and forged signatures!")

        for w in all_writers[:3]:
            print(f"  Writer {w}: {len(self.genuine_by_writer[w])} genuine, {len(self.forged_by_writer[w])} forged")

        split_idx = int(len(all_writers) * train_ratio)
        self.writers = all_writers[:split_idx] if split == "train" else all_writers[split_idx:]

        # Writers usable for pair sampling (need >=2 genuine, >=1 forged)
        self.usable = [w for w in self.writers
                       if len(self.genuine_by_writer.get(w, [])) >= 2
                       and len(self.forged_by_writer.get(w, [])) >= 1]

        # TRAIN: pairs are sampled fresh on every __getitem__, so each
        # epoch sees different combinations. A fixed pair list reuses the
        # same ~48k pairs every epoch out of ~680k possible ones, which
        # is a fast route to memorizing the training writers (observed:
        # train acc 0.97 vs val acc 0.80).
        # VAL: fixed, seeded pairs so scores are comparable across runs.
        self.dynamic = (split == "train")
        self.epoch_size = len(self.usable) * 60
        if self.dynamic:
            self.pairs = []
            print(f"[Dataset] {split}: {len(self.writers)} writers, "
                  f"{self.epoch_size} pairs/epoch (resampled every epoch)")
        else:
            self.pairs = self._generate_pairs(rng=random.Random(4242))
            print(f"[Dataset] {split}: {len(self.writers)} writers, "
                  f"{len(self.pairs)} pairs (fixed)")

    def _generate_pairs(self, pairs_per_writer=60, rng=None):
        """
        Build the fixed (validation) pair list. Also records the writer
        behind each pair in self.pair_writers, so evaluation can report
        metrics per source dataset WITHOUT re-splitting — re-splitting a
        single dataset would put writers the model trained on into its
        "validation" set and inflate the numbers.
        """
        rng = rng or random
        tagged = []
        for writer in self.usable:
            genuine = self.genuine_by_writer[writer]
            forged = self.forged_by_writer[writer]
            for _ in range(pairs_per_writer // 2):
                g1, g2 = rng.sample(genuine, 2)
                tagged.append(((g1, g2, 0), writer))
            for _ in range(pairs_per_writer // 2):
                tagged.append((
                    (rng.choice(genuine), rng.choice(forged), 1), writer))
        rng.shuffle(tagged)
        self.pair_writers = [w for _, w in tagged]
        return [p for p, _ in tagged]

    @property
    def rng(self):
        """
        Per-worker RNG. DataLoader workers each receive a *copy* of this
        object, so a shared pre-seeded Random would make every worker
        draw the identical sequence. Seed lazily, once, per process.
        """
        if not getattr(self, "_rng_ready", False):
            info = torch.utils.data.get_worker_info()
            seed = torch.initial_seed() + (info.id if info else 0)
            self._rng = random.Random(seed)
            self._rng_ready = True
        return self._rng

    def _sample_pair(self):
        """Draw one random pair — fresh combinations every epoch."""
        rng = self.rng
        writer = rng.choice(self.usable)
        genuine = self.genuine_by_writer[writer]
        if rng.random() < 0.5:
            g1, g2 = rng.sample(genuine, 2)
            return g1, g2, 0
        return rng.choice(genuine), rng.choice(self.forged_by_writer[writer]), 1

    def __len__(self):
        return self.epoch_size if self.dynamic else len(self.pairs)

    def __getitem__(self, idx):
        if self.dynamic:
            path1, path2, label = self._sample_pair()
        else:
            path1, path2, label = self.pairs[idx]
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)
        if self.augment:
            img1 = augment_signature(img1, self.rng)
            img2 = augment_signature(img2, self.rng)
        label = torch.tensor(label, dtype=torch.float32)
        return img1, img2, label


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path>")
        sys.exit(1)
    ds = SignaturePairDataset(sys.argv[1], split="train")
    print(f"\nTotal pairs: {len(ds)}")
    img1, img2, label = ds[0]
    print(f"img1: {img1.shape}, img2: {img2.shape}, label: {label}")
    print("Done!")