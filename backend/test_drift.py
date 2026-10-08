"""
Drift Analysis Test — Simulates backdated signature history for a user.

This populates signatures with realistic timestamps (spread over months)
so drift analysis can actually run on real data.

Three scenarios:
  1. AGING      → gradual similarity decline over 18 months (writer 1)
  2. MEDICAL    → stable, then sudden drop (writer 5)
  3. FORGERY    → rapid improvement over weeks (writer 10)

Run from backend folder:
  python test_drift.py --scenario aging
  python test_drift.py --scenario medical
  python test_drift.py --scenario forgery
"""
import sys
import argparse
import uuid
import random
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from app.config import SIGNATURES_DIR
from app.database.db import get_db, init_db
from app.models.embedding import extract_embedding
from app.services.signature_proc import extract_stroke_features
from app.services.drift_analyzer import analyze_drift, generate_explanation
from app.services.verification import _make_decision

DATASET = Path("../dataset/CEDAR")


def simulate_user(scenario: str):
    """Create a user with backdated signature history for the given scenario."""
    init_db()

    # Pick source writer based on scenario
    writer_map = {"aging": 1, "medical": 5, "forgery": 10, "normal": 15}
    writer_id = writer_map.get(scenario, 1)

    writer_dir = DATASET / str(writer_id)
    if not writer_dir.exists():
        print(f"❌ Writer {writer_id} not found at {writer_dir}")
        return

    genuines = sorted(writer_dir.glob("original_*.png"))
    forgeries = sorted(writer_dir.glob("forgeries_*.png"))

    if len(genuines) < 10:
        print(f"❌ Not enough signatures for writer {writer_id}")
        return

    # Create user
    user_id = str(uuid.uuid4())
    user_name = f"Simulated {scenario.title()} User"
    now = datetime.now()
    enrollment_date = (now - timedelta(days=550)).strftime("%Y-%m-%d")  # 18 months ago

    print(f"\n{'=' * 65}")
    print(f"  SCENARIO: {scenario.upper()}")
    print(f"  User: {user_name}")
    print(f"  Enrolled: {enrollment_date} ({(now - datetime.fromisoformat(enrollment_date)).days} days ago)")
    print(f"{'=' * 65}\n")

    # Copy reference signatures to user directory
    user_dir = SIGNATURES_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    ref_embeddings = []
    ref_paths = []
    for i in range(3):
        src = genuines[i]
        dst = user_dir / f"ref_{i+1}.png"
        shutil.copy2(src, dst)
        emb = extract_embedding(str(dst))
        ref_embeddings.append(emb)
        ref_paths.append(str(dst))

    baseline = np.mean(ref_embeddings, axis=0)
    baseline = baseline / (np.linalg.norm(baseline) + 1e-8)

    # ── Create user in DB ──
    with get_db() as db:
        db.execute(
            "INSERT INTO users (id, name, enrollment_date, age_at_enrollment, status) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_name, enrollment_date, 70, "active")
        )

        for i, (path, emb) in enumerate(zip(ref_paths, ref_embeddings)):
            features = extract_stroke_features(path)
            db.execute(
                """INSERT INTO signatures
                   (id, user_id, image_path, embedding, capture_date, is_reference,
                    similarity_score, pressure_mean, stroke_speed, tremor_index,
                    stroke_consistency, pen_lift_count)
                   VALUES (?, ?, ?, ?, ?, 1, 1.0, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), user_id, path, emb.tobytes(), enrollment_date,
                 features["pressure_mean"], features["stroke_speed"],
                 features["tremor_index"], features["stroke_consistency"],
                 features["pen_lift_count"])
            )

        db.execute(
            """INSERT INTO drift_profiles
               (id, user_id, drift_rate, drift_direction, volatility, trend_consistency,
                last_updated, baseline_embedding, expected_score, tenure_months, trust_buffer)
               VALUES (?, ?, 0.0, 'stable', 0.0, 1.0, ?, ?, 1.0, 18, 0.09)""",
            (str(uuid.uuid4()), user_id, now.isoformat(), baseline.tobytes())
        )

    # ── Generate synthetic scores over 18 months ──
    print("Generating signature history over 18 months...\n")
    score_timeline = generate_scenario_scores(scenario, num_signatures=15)

    scores = []
    dates = []
    tremors = []

    enroll_dt = datetime.fromisoformat(enrollment_date)

    for i, (days_offset, score, tremor) in enumerate(score_timeline):
        capture_date = (enroll_dt + timedelta(days=days_offset)).strftime("%Y-%m-%d")

        # Use a real signature (alternating genuine/forged for forgery scenario)
        if scenario == "forgery":
            # First 8 are forgery attempts (genuine-vs-forged similarity),
            # but scores are synthetic to simulate improving forger
            img_path = str(forgeries[i % len(forgeries)])
        else:
            img_path = str(genuines[(i + 3) % len(genuines)])

        # Copy signature
        dst = user_dir / f"sig_{capture_date}_{i}.png"
        shutil.copy2(img_path, dst)

        # Extract real features
        features = extract_stroke_features(str(dst))
        # Override tremor for scenario
        features["tremor_index"] = tremor

        # Use a synthetic embedding close to baseline with drift
        emb = extract_embedding(str(dst))

        sig_id = str(uuid.uuid4())
        with get_db() as db:
            db.execute(
                """INSERT INTO signatures
                   (id, user_id, image_path, embedding, capture_date, is_reference,
                    similarity_score, pressure_mean, stroke_speed, tremor_index,
                    stroke_consistency, pen_lift_count)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)""",
                (sig_id, user_id, str(dst), emb.tobytes(), capture_date, score,
                 features["pressure_mean"], features["stroke_speed"],
                 tremor, features["stroke_consistency"], features["pen_lift_count"])
            )

        scores.append(score)
        dates.append(capture_date)
        tremors.append(tremor)

        marker = "   "
        if scenario == "medical" and i == 10:
            marker = "🏥 "  # Medical event point
        if scenario == "forgery" and i >= 10:
            marker = "🕵️ "  # Forgery attempt
        print(f"  {marker}Day {days_offset:4d}  ({capture_date})  →  similarity={score:.3f}  tremor={tremor:.3f}")

    # ── Run drift analysis ──
    print(f"\n{'─' * 65}")
    print("  RUNNING DRIFT ANALYSIS")
    print(f"{'─' * 65}\n")

    profile = analyze_drift(
        scores=scores,
        dates=dates,
        tremor_values=tremors,
        enrollment_date=enrollment_date,
    )

    print(f"  Classification:       {profile.classification.upper()}")
    print(f"  Drift direction:      {profile.drift_direction}")
    print(f"  Drift rate/month:     {profile.drift_rate:+.4f}")
    print(f"  Volatility:           {profile.volatility:.4f}")
    print(f"  Trend consistency:    {profile.trend_consistency:.4f}  (R²)")
    print(f"  Expected score today: {profile.expected_score:.4f}")
    print(f"  Tenure:               {profile.tenure_months} months")
    print(f"  Trust buffer:         {profile.trust_buffer:.3f}")
    print(f"  Adjusted threshold:   {profile.adjusted_threshold:.3f}")
    print(f"  Confidence:           {profile.confidence:.3f}")
    print(f"\n  Details: {profile.details}")

    # Run the REAL decision engine, not a copy of it — otherwise the
    # simulation can report a verdict the live system would never give.
    mock_score = scores[-1]
    decision, _, effective_threshold = _make_decision(
        mock_score, profile.adjusted_threshold, profile)

    reason = generate_explanation(profile, mock_score, decision)

    print(f"\n{'─' * 65}")
    print(f"  FINAL DECISION: {decision.upper()}")
    print(f"  Score {mock_score:.3f} vs effective threshold {effective_threshold:.3f}", end="")
    print(f"  (aging allowance {profile.aging_allowance:.3f})"
          if profile.aging_allowance else "")
    print(f"{'─' * 65}")
    print(f"\n  Reason: {reason}\n")

    print(f"\n✅ User created. user_id: {user_id}")
    print(f"   View in API: http://localhost:8000/api/users/{user_id}")
    print(f"   History:     http://localhost:8000/api/users/{user_id}/history\n")


def generate_scenario_scores(scenario: str, num_signatures=15):
    """
    Returns list of (days_since_enrollment, score, tremor_index).
    """
    random.seed(42)
    timeline = []

    if scenario == "aging":
        # Slow linear decline: 0.96 → 0.81 over 550 days
        # Low volatility, consistent drift
        for i in range(num_signatures):
            day = int(20 + (i / num_signatures) * 500)
            base = 0.96 - (i / num_signatures) * 0.15
            noise = random.gauss(0, 0.015)
            score = max(0.5, min(1.0, base + noise))
            tremor = 0.15 + (i / num_signatures) * 0.10  # tremor increases with age
            timeline.append((day, round(score, 4), round(tremor, 4)))

    elif scenario == "medical":
        # Stable around 0.93 for 10 signatures, then sudden drop to 0.65-0.70
        for i in range(num_signatures):
            day = int(20 + (i / num_signatures) * 500)
            if i < 10:
                # Pre-event: stable
                score = 0.93 + random.gauss(0, 0.02)
                tremor = 0.18 + random.gauss(0, 0.01)
            else:
                # Post-event: sudden drop, high variance
                score = 0.68 + random.gauss(0, 0.06)
                tremor = 0.35 + random.gauss(0, 0.04)  # tremor jumps after stroke
            timeline.append((day, round(max(0.4, min(1.0, score)), 4), round(max(0.05, tremor), 4)))

    elif scenario == "forgery":
        # First 8: genuine user (high scores, normal tremor)
        # Last 7: forger practicing (scores rising rapidly, abnormally LOW tremor)
        for i in range(num_signatures):
            day = int(20 + (i / num_signatures) * 500)
            if i < 8:
                score = 0.92 + random.gauss(0, 0.02)
                tremor = 0.20 + random.gauss(0, 0.02)
            else:
                # Forger attempts — rapid improvement, unnaturally smooth strokes
                forger_progress = (i - 7) / 7  # 0 to 1
                score = 0.45 + forger_progress * 0.35 + random.gauss(0, 0.02)
                tremor = 0.05 + random.gauss(0, 0.01)  # very low = drawn slowly
            timeline.append((day, round(max(0.3, min(1.0, score)), 4), round(max(0.02, tremor), 4)))

    elif scenario == "normal":
        # Stable, no drift
        for i in range(num_signatures):
            day = int(20 + (i / num_signatures) * 500)
            score = 0.92 + random.gauss(0, 0.02)
            tremor = 0.20 + random.gauss(0, 0.015)
            timeline.append((day, round(score, 4), round(tremor, 4)))

    return timeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["aging", "medical", "forgery", "normal", "all"],
        default="aging",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        for s in ["normal", "aging", "medical", "forgery"]:
            simulate_user(s)
    else:
        simulate_user(args.scenario)