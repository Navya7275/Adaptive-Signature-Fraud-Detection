"""
Signature image preprocessing and stroke feature extraction.

Pipeline:
  Raw image → Grayscale → Otsu binarization → Crop to signature region
  → Resize to 150x220 → Normalize → Tensor

Stroke features (estimated from the static image):
  - tremor_index: high-frequency contour variation
  - pressure_mean: average ink intensity (darker = more pressure)
  - stroke_speed: inverse of stroke density (thinner strokes = faster)
  - stroke_consistency: how uniform the stroke width is
  - pen_lift_count: number of disconnected components (proxy for pen lifts)
"""
import cv2
import numpy as np
import torch
from app.config import IMG_HEIGHT, IMG_WIDTH


def remove_ruled_lines(img: np.ndarray, binary: np.ndarray) -> tuple:
    """
    Remove ruled notebook lines and box borders from a signature photo.

    Detects long straight horizontal/vertical runs with morphological opening
    and whites them out in both the grayscale image and the binary mask.

    Only near-border-to-border runs (>= 90% of the image dimension) count as
    page ruling. Anything shorter is treated as signature ink: flourish
    underlines, sweeping tails, and the Devanagari shirorekha (headline
    stroke) are all long horizontal strokes that must survive. A stricter
    kernel (e.g. width/4) measurably deleted up to 30% of the ink in some
    training signatures.
    """
    h, w = binary.shape
    h_len = int(w * 0.9)   # a ruled line spans essentially the whole page
    v_len = int(h * 0.9)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )
    # Cover anti-aliased edges of the detected lines
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))

    img_clean = img.copy()
    img_clean[lines > 0] = 255
    binary_clean = binary.copy()
    binary_clean[lines > 0] = 0
    return img_clean, binary_clean


def load_and_binarize(image_path: str) -> tuple:
    """Load grayscale + Otsu binary, with ruled lines/borders removed."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img, binary = remove_ruled_lines(img, binary)
    return img, binary


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load a signature image and return a normalized tensor ready for the CNN.
    Returns: Tensor of shape (1, 150, 220)
    """
    img, binary = load_and_binarize(image_path)

    # Crop to signature bounding box
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add small padding
        pad = 10
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        img = img[y1:y2, x1:x2]

    # Resize maintaining aspect ratio, pad to target size
    img = resize_with_padding(img, IMG_HEIGHT, IMG_WIDTH)

    # Normalize to [0, 1] and INVERT: ink = 1, background = 0.
    # Signature images are ~95% background; with white-high polarity the
    # network's pooled features are dominated by the constant background
    # and every signature embeds nearly identically. Ink-high polarity
    # makes activations sparse and stroke-driven.
    img = 1.0 - (img.astype(np.float32) / 255.0)

    # To tensor: (1, H, W)
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


def resize_with_padding(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image to fit within target dimensions, pad with white."""
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas and center the signature
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_stroke_features(image_path: str) -> dict:
    """
    Extract stroke-level features from a signature image.
    These features help distinguish aging degradation from forgery.
    """
    img, binary = load_and_binarize(image_path)

    # ── Pressure Mean ──
    # Darker ink pixels = more pressure. We measure mean intensity of ink pixels.
    ink_pixels = img[binary > 0]
    pressure_mean = float(255 - np.mean(ink_pixels)) / 255.0 if len(ink_pixels) > 0 else 0.0

    # ── Tremor Index ──
    # Find contours, measure high-frequency variation in contour curvature.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tremor_index = compute_tremor(contours)

    # ── Stroke Speed (proxy) ──
    # Thinner strokes suggest faster writing. We use the ratio of
    # skeleton length to ink area.
    skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else thin_manual(binary)
    skel_pixels = np.count_nonzero(skeleton)
    ink_area = np.count_nonzero(binary)
    stroke_speed = float(skel_pixels) / max(ink_area, 1)

    # ── Stroke Consistency ──
    # How uniform is the stroke width? Low variance = consistent writing.
    if skel_pixels > 0:
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        widths = dist_transform[skeleton > 0]
        stroke_consistency = 1.0 / (1.0 + float(np.std(widths)))
    else:
        stroke_consistency = 0.0

    # ── Pen Lift Count ──
    # Number of connected components (minus background)
    num_labels, _ = cv2.connectedComponents(binary)
    pen_lift_count = max(0, num_labels - 1)

    return {
        "pressure_mean": round(pressure_mean, 4),
        "tremor_index": round(tremor_index, 4),
        "stroke_speed": round(stroke_speed, 4),
        "stroke_consistency": round(stroke_consistency, 4),
        "pen_lift_count": pen_lift_count,
    }


def compute_tremor(contours) -> float:
    """
    Measure tremor by analyzing high-frequency direction changes in contours.
    Aging hands produce more tremor (zig-zag patterns).
    Forgers drawing slowly produce unnaturally smooth strokes (low tremor).
    """
    if not contours:
        return 0.0

    total_changes = 0
    total_points = 0

    for contour in contours:
        if len(contour) < 10:
            continue
        pts = contour.squeeze()
        if pts.ndim != 2:
            continue

        # Compute direction angles between consecutive points
        diffs = np.diff(pts, axis=0).astype(np.float64)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])

        # Count rapid direction changes (high-frequency oscillation)
        angle_changes = np.abs(np.diff(angles))
        # Wrap to [-pi, pi]
        angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)

        # Tremor = proportion of sharp direction changes
        sharp_changes = np.sum(angle_changes > np.pi / 4)
        total_changes += sharp_changes
        total_points += len(angle_changes)

    if total_points == 0:
        return 0.0
    return float(total_changes) / total_points


def thin_manual(binary: np.ndarray) -> np.ndarray:
    """
    Fallback skeletonization if cv2.ximgproc is not available.
    Uses morphological thinning.
    """
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary.copy()

    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel