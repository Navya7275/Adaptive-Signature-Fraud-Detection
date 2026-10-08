"""
User Manager — enrollment, history retrieval, and profile management.
"""
import hashlib
import uuid
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

from app.config import SIGNATURES_DIR, BASE_THRESHOLD
from app.database.db import get_db, row_to_dict, rows_to_dicts
from app.models.embedding import extract_embedding, compare_embeddings
from app.services.signature_proc import extract_stroke_features


def compute_personal_threshold(embeddings: list) -> float:
    """
    Calibrate a per-user verification threshold from how consistent the
    user's own reference signatures are with each other.

    A signer whose references agree tightly gets a stricter threshold;
    a naturally variable signer gets a fairer, lower one. Falls back to
    the global BASE_THRESHOLD when there aren't enough references.
    """
    if len(embeddings) < 2:
        return BASE_THRESHOLD

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(compare_embeddings(embeddings[i], embeddings[j]))

    mu = float(np.mean(sims))
    sigma = float(np.std(sims))
    # 2 sigma below the user's own consistency, small safety margin,
    # clamped to the trained model's score space (genuine ~0.94,
    # skilled forgeries ~0.80 — a threshold under 0.84 admits forgeries,
    # over 0.95 rejects the user's own natural variation).
    threshold = mu - 2.0 * sigma - 0.01
    return float(np.clip(threshold, 0.84, 0.95))


def enroll_user(
    name: str,
    age: int,
    reference_image_paths: list[str],
) -> dict:
    """
    Enroll a new user with their reference signatures.

    Args:
        name: user's full name
        age: age at enrollment
        reference_image_paths: list of paths to reference signature images (3-5 recommended)

    Returns:
        dict with user_id and enrollment details
    """
    if len(reference_image_paths) < 1:
        return {"error": "At least 1 reference signature is required (3-5 recommended)"}

    user_id = str(uuid.uuid4())
    now = datetime.now()
    enrollment_date = now.strftime("%Y-%m-%d")

    # Create user-specific directory for signatures
    user_dir = SIGNATURES_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # Process each reference signature
    embeddings = []
    sig_records = []

    for i, src_path in enumerate(reference_image_paths):
        # Copy signature to user directory
        ext = Path(src_path).suffix or ".png"
        dest_path = user_dir / f"ref_{i+1}{ext}"
        shutil.copy2(src_path, dest_path)

        # Extract embedding
        emb = extract_embedding(str(dest_path))
        embeddings.append(emb)

        # Extract stroke features
        features = extract_stroke_features(str(dest_path))

        sig_id = str(uuid.uuid4())
        sig_records.append({
            "id": sig_id,
            "user_id": user_id,
            "image_path": str(dest_path),
            "file_hash": hashlib.md5(dest_path.read_bytes()).hexdigest(),
            "embedding": emb.tobytes(),
            "capture_date": enrollment_date,
            "is_reference": 1,
            "similarity_score": 1.0,  # reference = perfect match to itself
            **features,
        })

    # Compute baseline embedding (average of all reference embeddings)
    baseline_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    # Re-normalize
    baseline_embedding = baseline_embedding / (np.linalg.norm(baseline_embedding) + 1e-8)

    # Per-user calibrated threshold from reference consistency
    personal_threshold = compute_personal_threshold(embeddings)

    # Save to database
    with get_db() as db:
        db.execute(
            "INSERT INTO users (id, name, enrollment_date, age_at_enrollment, status) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, enrollment_date, age, "active")
        )

        for sig in sig_records:
            db.execute(
                """INSERT INTO signatures
                   (id, user_id, image_path, file_hash, embedding, capture_date,
                    is_reference, similarity_score, pressure_mean, stroke_speed,
                    tremor_index, stroke_consistency, pen_lift_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sig["id"], sig["user_id"], sig["image_path"], sig["file_hash"],
                    sig["embedding"], sig["capture_date"], sig["is_reference"],
                    sig["similarity_score"], sig["pressure_mean"], sig["stroke_speed"],
                    sig["tremor_index"], sig["stroke_consistency"], sig["pen_lift_count"],
                )
            )

        # Create initial drift profile
        db.execute(
            """INSERT INTO drift_profiles
               (id, user_id, drift_rate, drift_direction, volatility, trend_consistency,
                last_updated, baseline_embedding, expected_score, tenure_months,
                trust_buffer, base_threshold)
               VALUES (?, ?, 0.0, 'stable', 0.0, 1.0, ?, ?, 1.0, 0, 0.0, ?)""",
            (str(uuid.uuid4()), user_id, now.isoformat(),
             baseline_embedding.tobytes(), personal_threshold)
        )

    return {
        "user_id": user_id,
        "name": name,
        "age": age,
        "enrollment_date": enrollment_date,
        "reference_signatures": len(reference_image_paths),
        "personal_threshold": round(personal_threshold, 4),
        "message": f"User enrolled successfully with {len(reference_image_paths)} reference signatures",
    }


def get_user(user_id: str) -> dict:
    """Get user details with drift profile."""
    with get_db() as db:
        user = row_to_dict(db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone())
        if not user:
            return None

        profile = row_to_dict(db.execute(
            "SELECT * FROM drift_profiles WHERE user_id = ?", (user_id,)
        ).fetchone())

        sig_count = db.execute(
            "SELECT COUNT(*) as count FROM signatures WHERE user_id = ?", (user_id,)
        ).fetchone()["count"]

        recent_logs = rows_to_dicts(db.execute(
            """SELECT decision, drift_classification, raw_score, timestamp
               FROM verification_logs WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT 10""",
            (user_id,)
        ).fetchall())

    result = {**user, "signature_count": sig_count, "recent_verifications": recent_logs}
    if profile:
        result["drift_profile"] = {
            "drift_rate": profile["drift_rate"],
            "drift_direction": profile["drift_direction"],
            "volatility": profile["volatility"],
            "trend_consistency": profile["trend_consistency"],
            "expected_score": profile["expected_score"],
            "tenure_months": profile["tenure_months"],
            "trust_buffer": profile["trust_buffer"],
            "base_threshold": profile.get("base_threshold", 0.80),
        }
    return result


def get_user_history(user_id: str) -> dict:
    """Get full signature and verification history for visualization."""
    with get_db() as db:
        signatures = rows_to_dicts(db.execute(
            """SELECT id, capture_date, is_reference, similarity_score,
                      pressure_mean, stroke_speed, tremor_index,
                      stroke_consistency, pen_lift_count
               FROM signatures WHERE user_id = ? ORDER BY capture_date ASC""",
            (user_id,)
        ).fetchall())

        logs = rows_to_dicts(db.execute(
            """SELECT timestamp, raw_score, adjusted_score, threshold_used,
                      decision, drift_classification, confidence, reason
               FROM verification_logs WHERE user_id = ?
               ORDER BY timestamp ASC""",
            (user_id,)
        ).fetchall())

        alerts = rows_to_dicts(db.execute(
            "SELECT * FROM alerts WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall())

    return {
        "signatures": signatures,
        "verification_logs": logs,
        "alerts": alerts,
    }


def list_users() -> list:
    """List all enrolled users with summary info."""
    with get_db() as db:
        users = rows_to_dicts(db.execute(
            """SELECT u.id, u.name, u.enrollment_date, u.age_at_enrollment, u.status,
                      dp.drift_direction, dp.drift_rate, dp.tenure_months, dp.trust_buffer,
                      COUNT(s.id) as sig_count
               FROM users u
               LEFT JOIN drift_profiles dp ON u.id = dp.user_id
               LEFT JOIN signatures s ON u.id = s.user_id
               GROUP BY u.id
               ORDER BY u.created_at DESC""",
        ).fetchall())
    return users