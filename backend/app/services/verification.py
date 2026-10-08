"""
Adaptive Verification Engine — the decision layer.

This ties everything together:
  1. Takes a new signature image + user ID
  2. Extracts embedding and stroke features
  3. Compares against the user's adaptive baseline
  4. Runs drift analysis on their history
  5. Makes a three-way decision: approved / escalated / rejected
  6. Generates a plain-language explanation
  7. Logs everything and updates the drift profile
"""
import hashlib
import shutil
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path

from app.config import (
    BASE_THRESHOLD, ESCALATION_BAND, SIGNATURES_DIR, AGING_MAX_ALLOWANCE,
)
from app.database.db import get_db, row_to_dict, rows_to_dicts
from app.models.embedding import (
    extract_embedding, compare_embeddings, compare_embedding_to_image,
)
from app.services.signature_proc import extract_stroke_features
from app.services.drift_analyzer import (
    analyze_drift, generate_explanation, DriftProfile,
)


def verify_signature(user_id: str, image_path: str) -> dict:
    """
    Main verification pipeline.

    Returns a dict with:
      - decision: "approved" | "escalated" | "rejected"
      - raw_score, adjusted_score, threshold_used
      - drift_classification
      - confidence
      - reason (plain-language explanation)
      - profile (full drift profile details)
    """
    now = datetime.now().isoformat()

    # Fingerprint the upload for duplicate/replay detection
    file_hash = hashlib.md5(Path(image_path).read_bytes()).hexdigest()

    # ── Step 1: Get user and their history ──
    with get_db() as db:
        user = row_to_dict(db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone())

        if not user:
            return {"error": f"User {user_id} not found"}

        if user["status"] == "suspended":
            return {"error": "User account is suspended", "decision": "rejected"}

        # ── Duplicate / replay detection ──
        # The exact same image file being submitted again is either a
        # testing mistake or a replay attack (e.g. a photocopied capture).
        # Real signatures are never pixel-identical twice.
        dup = db.execute(
            "SELECT is_reference FROM signatures WHERE user_id = ? AND file_hash = ?",
            (user_id, file_hash)
        ).fetchone()
        if dup:
            kind = "an enrollment reference" if dup["is_reference"] else "a previously verified signature"
            return {"error":
                    f"Duplicate image: this exact file is already stored for this "
                    f"user as {kind}. Every real signing produces a unique image — "
                    f"please upload a different photo."}

        # Get drift profile
        drift_row = row_to_dict(db.execute(
            "SELECT * FROM drift_profiles WHERE user_id = ?", (user_id,)
        ).fetchone())

        # Get signature history (chronological)
        sig_rows = rows_to_dicts(db.execute(
            "SELECT * FROM signatures WHERE user_id = ? ORDER BY capture_date ASC",
            (user_id,)
        ).fetchall())

    # ── Step 2: Extract embedding + features from new signature ──
    new_embedding = extract_embedding(image_path)
    stroke_features = extract_stroke_features(image_path)

    # ── Step 3: Compute similarity score (multi-reference) ──
    # Compare against the adaptive baseline AND each reference signature,
    # then take the mean of the top-2 matches. More robust than a single
    # averaged baseline: a variable signer matches at least some references.
    candidate_scores = []

    if drift_row and drift_row.get("baseline_embedding"):
        baseline_emb = np.frombuffer(drift_row["baseline_embedding"], dtype=np.float32)
        candidate_scores.append(compare_embeddings(baseline_emb, new_embedding))

    ref_sigs = [s for s in sig_rows if s.get("is_reference")]
    if not ref_sigs:
        ref_sigs = sig_rows[:3]  # use earliest signatures
    for ref in ref_sigs:
        if ref.get("embedding"):
            ref_emb = np.frombuffer(ref["embedding"], dtype=np.float32)
            candidate_scores.append(compare_embeddings(ref_emb, new_embedding))

    if not candidate_scores:
        return {"error": "No reference signatures found for this user"}

    top2 = sorted(candidate_scores, reverse=True)[:2]
    raw_score = float(np.mean(top2))

    # ── Step 4: Run drift analysis ──
    # Exclude reference signatures: they carry an artificial
    # similarity_score of 1.0 (self-match at enrollment), which would
    # anchor the regression at "perfect on day zero" and make every real
    # score look like decline — falsely triggering aging/medical patterns.
    verif_rows = [
        s for s in sig_rows
        if not s.get("is_reference") and s.get("similarity_score") is not None
    ]
    history_scores = [s["similarity_score"] for s in verif_rows]
    history_dates = [s["capture_date"] for s in verif_rows]
    history_tremors = [s.get("tremor_index", 0.0) for s in verif_rows]

    # Append current signature to history for analysis
    history_scores.append(raw_score)
    history_dates.append(now[:10])
    history_tremors.append(stroke_features.get("tremor_index", 0.0))

    user_base_threshold = BASE_THRESHOLD
    if drift_row and drift_row.get("base_threshold"):
        user_base_threshold = float(drift_row["base_threshold"])

    profile = analyze_drift(
        scores=history_scores,
        dates=history_dates,
        tremor_values=history_tremors,
        enrollment_date=user["enrollment_date"],
        base_threshold=user_base_threshold,
    )

    # ── Step 5: Make decision ──
    decision, adjusted_score, threshold = _make_decision(
        raw_score, profile.adjusted_threshold, profile)

    # ── Step 6: Generate explanation ──
    reason = generate_explanation(profile, raw_score, decision)

    # ── Step 7: Save signature, log, and update profile ──
    sig_id = str(uuid.uuid4())
    log_id = str(uuid.uuid4())

    # Persist the image into the user's folder — the uploaded file is a
    # temp file the route deletes after this call, so storing its path
    # would leave a dead reference in the database.
    user_dir = SIGNATURES_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(image_path).suffix or ".png"
    stored_path = user_dir / f"sig_{now[:10]}_{sig_id[:8]}{ext}"
    shutil.copy2(image_path, stored_path)

    with get_db() as db:
        # Save the new signature record
        db.execute(
            """INSERT INTO signatures
               (id, user_id, image_path, file_hash, embedding, capture_date, similarity_score,
                pressure_mean, stroke_speed, tremor_index, stroke_consistency, pen_lift_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sig_id, user_id, str(stored_path), file_hash,
                new_embedding.tobytes(), now[:10], raw_score,
                stroke_features["pressure_mean"],
                stroke_features["stroke_speed"],
                stroke_features["tremor_index"],
                stroke_features["stroke_consistency"],
                stroke_features["pen_lift_count"],
            )
        )

        # Save verification log
        db.execute(
            """INSERT INTO verification_logs
               (id, user_id, signature_id, timestamp, raw_score, adjusted_score,
                threshold_used, decision, drift_classification, confidence, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                log_id, user_id, sig_id, now, raw_score, adjusted_score,
                threshold, decision, profile.classification, profile.confidence, reason,
            )
        )

        # Update drift profile
        _update_drift_profile(db, user_id, profile, new_embedding, decision)

        # Create alert if needed
        if profile.classification == "forgery_attempt":
            db.execute(
                """INSERT INTO alerts (id, user_id, alert_type, severity, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), user_id, "forgery_suspected", "critical",
                 f"Forgery pattern detected. Score: {raw_score:.3f}, Confidence: {profile.confidence:.2f}")
            )
        elif profile.classification == "medical_event":
            db.execute(
                """INSERT INTO alerts (id, user_id, alert_type, severity, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), user_id, "medical_event", "high",
                 f"Possible medical event detected. Score dropped significantly. Human review recommended.")
            )

    return {
        "decision": decision,
        "raw_score": round(raw_score, 4),
        "adjusted_score": round(adjusted_score, 4),
        "threshold_used": round(threshold, 4),
        "drift_classification": profile.classification,
        "drift_direction": profile.drift_direction,
        "confidence": round(profile.confidence, 4),
        "tenure_months": profile.tenure_months,
        "trust_buffer": round(profile.trust_buffer, 4),
        "aging_allowance": round(profile.aging_allowance, 4),
        "reason": reason,
        "stroke_features": stroke_features,
        "drift_details": profile.details,
        "signature_id": sig_id,
        "log_id": log_id,
    }


def _make_decision(raw_score: float, threshold: float,
                   profile: DriftProfile) -> tuple:
    """
    Three-way decision logic. Returns (decision, adjusted_score, threshold):
      - approved: score >= effective threshold AND no forgery pattern
      - escalated: score in escalation band OR medical event detected
      - rejected: score below (threshold - escalation band) OR forgery detected
    """
    # Override: forgery detected → always reject
    if profile.classification == "forgery_attempt" and profile.confidence > 0.6:
        return "rejected", raw_score, threshold

    # Override: medical event → always escalate
    if profile.classification == "medical_event":
        return "escalated", raw_score, threshold

    # ── Aging adaptation ──
    # A signature that drifts slowly, consistently and with low volatility
    # is aging, not fraud. Judge it against where the trend predicts it
    # should be today rather than against the enrollment-era bar, bounded
    # by AGING_MAX_ALLOWANCE so the threshold cannot erode without limit.
    effective = threshold
    if profile.classification == "natural_aging" and profile.confidence >= 0.5:
        predicted = profile.expected_score - 2.0 * max(profile.volatility, 0.01)
        floor = threshold - AGING_MAX_ALLOWANCE
        effective = max(floor, min(threshold, predicted))
        profile.aging_allowance = round(threshold - effective, 4)

    # Score-based decision
    if raw_score >= effective:
        return "approved", raw_score, effective

    elif raw_score >= (effective - ESCALATION_BAND):
        # In the escalation band — not clearly pass or fail
        return "escalated", raw_score, effective

    else:
        return "rejected", raw_score, effective


def _update_drift_profile(db, user_id: str, profile: DriftProfile,
                           new_embedding: np.ndarray, decision: str):
    """
    Update or create the user's drift profile in the database.

    The baseline embedding adapts ONLY on approved verifications, and only
    by a small blend (EMA) — never a full replacement. This lets the
    baseline follow genuine slow drift (aging) while preventing a forger
    from walking the template toward their own signature through a few
    borderline acceptances.
    """
    now = datetime.now().isoformat()

    existing = db.execute(
        "SELECT id, baseline_embedding FROM drift_profiles WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if existing:
        if decision == "approved" and existing["baseline_embedding"]:
            old = np.frombuffer(existing["baseline_embedding"], dtype=np.float32)
            blended = 0.9 * old + 0.1 * new_embedding
            blended = (blended / (np.linalg.norm(blended) + 1e-8)).astype(np.float32)
            baseline_bytes = blended.tobytes()
        else:
            baseline_bytes = existing["baseline_embedding"]

        db.execute(
            """UPDATE drift_profiles SET
               drift_rate=?, drift_direction=?, volatility=?, trend_consistency=?,
               last_updated=?, baseline_embedding=?, expected_score=?,
               tenure_months=?, trust_buffer=?
               WHERE user_id=?""",
            (
                profile.drift_rate, profile.drift_direction, profile.volatility,
                profile.trend_consistency, now, baseline_bytes,
                profile.expected_score, profile.tenure_months, profile.trust_buffer,
                user_id,
            )
        )
    else:
        # No profile yet (unusual — enrollment creates one). Seed the
        # baseline only from an APPROVED signature; a rejected forgery
        # must never become the user's identity template.
        baseline_bytes = new_embedding.tobytes() if decision == "approved" else None
        db.execute(
            """INSERT INTO drift_profiles
               (id, user_id, drift_rate, drift_direction, volatility, trend_consistency,
                last_updated, baseline_embedding, expected_score, tenure_months, trust_buffer)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), user_id, profile.drift_rate, profile.drift_direction,
                profile.volatility, profile.trend_consistency, now,
                baseline_bytes, profile.expected_score,
                profile.tenure_months, profile.trust_buffer,
            )
        )