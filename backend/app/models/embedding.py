"""
Embedding extraction pipeline.
Loads the trained Siamese encoder and provides a clean interface
for extracting 128-dim embeddings from signature images.
"""
import torch
import numpy as np
from app.config import MODEL_PATH, EMBEDDING_DIM
from app.models.siamese_net import SiameseNetwork, cosine_similarity_score
from app.services.signature_proc import preprocess_image


# Global model instance (loaded once)
_model = None
_device = None


def get_device():
    """Get the best available device."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"[Model] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu")
            print("[Model] Using CPU (GPU not available)")
    return _device


def load_model():
    """Load the trained Siamese model (lazy loading, singleton)."""
    global _model
    if _model is not None:
        return _model

    device = get_device()
    _model = SiameseNetwork(EMBEDDING_DIM)

    if MODEL_PATH.exists():
        state = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
        _model.load_state_dict(state)
        print(f"[Model] Loaded weights from {MODEL_PATH}")
    else:
        print(f"[Model] WARNING: No trained model found at {MODEL_PATH}")
        print(f"[Model] Using randomly initialized weights — train the model first!")

    _model = _model.to(device)
    _model.eval()
    return _model


def extract_embedding(image_path: str) -> np.ndarray:
    """
    Extract a 128-dim embedding from a signature image.

    Uses test-time augmentation: the image is embedded at small
    rotations as well and the embeddings are averaged (then
    re-normalized). This washes out capture-angle sensitivity in
    phone photos at the cost of a few extra forward passes.

    Returns: numpy array of shape (128,) as float32
    """
    import cv2

    model = load_model()
    device = get_device()

    tensor = preprocess_image(image_path)   # (1, H, W), ink-high
    img = tensor.squeeze(0).numpy()
    h, w = img.shape

    variants = [img]
    for angle in (-2.0, 2.0):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       flags=cv2.INTER_LINEAR, borderValue=0.0))

    batch = torch.from_numpy(np.stack(variants)).unsqueeze(1).to(device)

    with torch.no_grad():
        embs = model.get_embedding(batch)          # (3, 128), each L2-normed
        mean_emb = embs.mean(dim=0)
        mean_emb = mean_emb / (mean_emb.norm() + 1e-8)

    # Always return float32 for consistent byte size (128 * 4 = 512 bytes)
    return mean_emb.cpu().numpy().flatten().astype(np.float32)


def compare_signatures(image_path_1: str, image_path_2: str) -> float:
    """
    Compare two signature images and return similarity score [0, 1].
    """
    model = load_model()
    device = get_device()

    t1 = preprocess_image(image_path_1).unsqueeze(0).to(device)
    t2 = preprocess_image(image_path_2).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = model.get_embedding(t1)
        emb2 = model.get_embedding(t2)

    return cosine_similarity_score(emb1, emb2)


def compare_embedding_to_image(embedding: np.ndarray, image_path: str) -> float:
    """
    Compare a stored embedding against a new signature image.
    """
    model = load_model()
    device = get_device()

    # Explicit copy: frombuffer arrays are read-only, and
    # ascontiguousarray returns them unchanged (no copy) — torch warns.
    emb_arr = np.array(embedding, dtype=np.float32)
    stored = torch.from_numpy(emb_arr).unsqueeze(0).to(device)
    new_tensor = preprocess_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        new_emb = model.get_embedding(new_tensor)

    return cosine_similarity_score(stored, new_emb)


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compare two stored embeddings."""
    device = get_device()
    # Explicit copies: frombuffer arrays are read-only (torch warns)
    a1 = np.array(emb1, dtype=np.float32)
    a2 = np.array(emb2, dtype=np.float32)
    t1 = torch.from_numpy(a1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(a2).unsqueeze(0).to(device)
    return cosine_similarity_score(t1, t2)