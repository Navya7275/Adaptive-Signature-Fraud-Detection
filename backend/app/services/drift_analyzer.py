"""
Drift Analyzer — the statistical brain of the system.

This module does NOT use ML. It uses classical statistics to:
1. Track how each user's signature changes over time
2. Compute drift rate (slope), volatility (std), and trend consistency (R²)
3. Classify the drift pattern into one of four categories:
   - normal: stable, no significant drift
   - natural_aging: slow, consistent decline over months/years
   - medical_event: sudden drop followed by stabilization
   - forgery_attempt: rapid improvement over days/weeks

The classification logic:
  ┌─────────────────────────────────────────────────┐
  │ Compute slope of similarity scores over time     │
  │ Compute volatility (rolling std)                 │
  │ Compute short-term vs long-term trend            │
  ├─────────────────────────────────────────────────┤
  │ IF sudden drop + high volatility → MEDICAL       │
  │ IF improving + low tremor + no prior dip → FORGE │
  │ IF slow decline + consistent + low vol → AGING   │
  │ ELSE → NORMAL                                    │
  └─────────────────────────────────────────────────┘
"""
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from dataclasses import dataclass
from app.config import (
    MIN_SAMPLES_FOR_DRIFT, AGING_DRIFT_MAX_SLOPE, FORGERY_IMPROVEMENT_SLOPE,
    MEDICAL_EVENT_VOLATILITY, MEDICAL_EVENT_DROP, SHORT_WINDOW, LONG_WINDOW,
    TENURE_BUFFER_RATE, MAX_TRUST_BUFFER, BASE_THRESHOLD, ESCALATION_BAND,
)


@dataclass
class DriftProfile:
    drift_rate: float = 0.0
    drift_direction: str = "stable"      # degrading, improving, stable
    volatility: float = 0.0
    trend_consistency: float = 1.0       # R² value
    classification: str = "normal"       # normal, natural_aging, medical_event, forgery_attempt
    expected_score: float = 1.0
    tenure_months: int = 0
    trust_buffer: float = 0.0
    adjusted_threshold: float = BASE_THRESHOLD
    aging_allowance: float = 0.0         # extra relaxation granted for aging
    confidence: float = 0.0
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def analyze_drift(
    scores: list[float],
    dates: list[str],
    tremor_values: list[float],
    enrollment_date: str,
    base_threshold: float = BASE_THRESHOLD,
) -> DriftProfile:
    """
    Main entry point. Analyzes a user's signature history and returns
    a complete drift profile with classification.

    Args:
        scores: list of similarity scores (chronological order)
        dates: list of ISO date strings corresponding to each score
        tremor_values: list of tremor index values for each signature
        enrollment_date: ISO date string of when user enrolled
        base_threshold: per-user calibrated threshold (global default if absent)
    """
    profile = DriftProfile()

    # ── Tenure Calculation ──
    enroll_dt = datetime.fromisoformat(enrollment_date)
    now = datetime.now()
    profile.tenure_months = max(0, (now.year - enroll_dt.year) * 12 + now.month - enroll_dt.month)

    # ── Trust Buffer from Tenure ──
    profile.trust_buffer = min(
        profile.tenure_months * TENURE_BUFFER_RATE,
        MAX_TRUST_BUFFER
    )
    profile.adjusted_threshold = max(0.1, base_threshold - profile.trust_buffer)

    # Not enough data for drift analysis
    if len(scores) < MIN_SAMPLES_FOR_DRIFT:
        profile.expected_score = np.mean(scores) if scores else 1.0
        profile.confidence = 0.3
        profile.details["note"] = f"Only {len(scores)} samples — need {MIN_SAMPLES_FOR_DRIFT} for drift analysis"
        return profile

    scores_arr = np.array(scores, dtype=np.float64)
    tremor_arr = np.array(tremor_values, dtype=np.float64) if tremor_values else np.zeros(len(scores))

    # Convert dates to numeric (days since enrollment)
    date_dts = [datetime.fromisoformat(d) for d in dates]
    days_arr = np.array([(d - enroll_dt).days for d in date_dts], dtype=np.float64)

    # ── Linear Regression on Scores ──
    # Guard against degenerate case (all dates identical → no time axis)
    if np.std(days_arr) < 1e-6:
        # All signatures captured on the same day — no drift can be measured
        profile.expected_score = float(np.mean(scores_arr))
        profile.drift_rate = 0.0
        profile.drift_direction = "stable"
        profile.trend_consistency = 0.0
        profile.volatility = float(np.std(scores_arr))
        profile.confidence = 0.5
        profile.classification = "normal"
        profile.details = {
            "note": "All signatures from same date — drift cannot be computed yet",
            "num_samples": len(scores),
        }
        return profile

    slope, intercept, r_value, p_value, std_err = stats.linregress(days_arr, scores_arr)
    r_squared = r_value ** 2

    profile.drift_rate = float(slope * 30)  # convert to per-month
    profile.trend_consistency = float(r_squared)

    if slope < -0.0001:
        profile.drift_direction = "degrading"
    elif slope > 0.0001:
        profile.drift_direction = "improving"
    else:
        profile.drift_direction = "stable"

    # ── Volatility (rolling std) ──
    if len(scores_arr) >= 3:
        window = min(5, len(scores_arr))
        rolling_stds = []
        for i in range(window - 1, len(scores_arr)):
            chunk = scores_arr[i - window + 1:i + 1]
            rolling_stds.append(np.std(chunk))
        profile.volatility = float(np.mean(rolling_stds))
    else:
        profile.volatility = float(np.std(scores_arr))

    # ── Expected Score Today ──
    days_since_enroll = (now - enroll_dt).days
    profile.expected_score = float(np.clip(intercept + slope * days_since_enroll, 0, 1))

    # ── Short-term vs Long-term Trend ──
    short_trend = _compute_short_trend(scores_arr)
    long_trend = _compute_long_trend(scores_arr)

    # ── Detect Sudden Discontinuity (Medical Event) ──
    # Look for a sharp change-point in the score series, not just recent vs old.
    discontinuity = _detect_discontinuity(scores_arr)

    # ── Tremor Jump Detection ──
    # Medical events (stroke, neuropathy) cause sudden tremor spikes.
    tremor_jump = 0.0
    if len(tremor_arr) >= SHORT_WINDOW + 3:
        pre = np.mean(tremor_arr[:-SHORT_WINDOW])
        post = np.mean(tremor_arr[-SHORT_WINDOW:])
        tremor_jump = post - pre

    # ── Recent Score Drop Detection ──
    recent_drop = _detect_recent_drop(scores_arr)

    # ── Average Tremor ──
    recent_tremor = float(np.mean(tremor_arr[-SHORT_WINDOW:])) if len(tremor_arr) > 0 else 0.0
    historical_tremor = float(np.mean(tremor_arr)) if len(tremor_arr) > 0 else 0.0

    # ═══════════════════════════════════════
    #         DRIFT CLASSIFICATION
    # ═══════════════════════════════════════

    profile.details = {
        "slope_per_month": round(profile.drift_rate, 6),
        "r_squared": round(r_squared, 4),
        "volatility": round(profile.volatility, 4),
        "short_trend": round(short_trend, 6),
        "long_trend": round(long_trend, 6),
        "recent_drop": round(recent_drop, 4),
        "discontinuity": round(discontinuity, 4),
        "recent_tremor": round(recent_tremor, 4),
        "historical_tremor": round(historical_tremor, 4),
        "tremor_jump": round(tremor_jump, 4),
        "num_samples": len(scores),
    }

    # ── Signal extraction for rule decisions ──
    # Forgery fingerprint: recent scores rising + tremor dropped significantly.
    # (A forger practices → scores improve, but they draw slowly → tremor is unnaturally low.)
    tremor_ratio = (recent_tremor / historical_tremor) if historical_tremor > 0 else 1.0
    forgery_fingerprint = (short_trend > FORGERY_IMPROVEMENT_SLOPE and
                           tremor_jump < -0.05 and
                           tremor_ratio < 0.7)

    # Medical fingerprint: scores dropped AND tremor increased.
    # (A medical event causes loss of motor control → more tremor.)
    medical_fingerprint = (discontinuity > MEDICAL_EVENT_DROP and tremor_jump > 0.05)

    # Rule 1: FORGERY ATTEMPT (check FIRST — forgeries can look like medical drops)
    # Key distinguishing feature: tremor DROPS in forgery (forger draws slowly),
    # but tremor RISES in medical events (loss of motor control).
    if forgery_fingerprint:
        profile.classification = "forgery_attempt"
        profile.confidence = min(0.95, 0.55 + abs(short_trend) * 8 + (1 - tremor_ratio) * 0.3)
        return profile

    # Very fast improvement alone is suspicious even without tremor signal
    if short_trend > FORGERY_IMPROVEMENT_SLOPE * 2 and tremor_jump <= 0:
        profile.classification = "forgery_attempt"
        profile.confidence = min(0.9, 0.45 + abs(short_trend) * 8)
        return profile

    # Rule 2: MEDICAL EVENT
    # Sudden discontinuity + tremor increase, OR large drop + high volatility
    if medical_fingerprint:
        profile.classification = "medical_event"
        profile.confidence = min(0.95, 0.55 + discontinuity + tremor_jump)
        return profile

    if (recent_drop > MEDICAL_EVENT_DROP and
            profile.volatility > MEDICAL_EVENT_VOLATILITY and
            tremor_jump >= 0):  # tremor not decreasing
        profile.classification = "medical_event"
        profile.confidence = min(0.95, 0.5 + recent_drop + profile.volatility)
        return profile

    # Rule 3: NATURAL AGING
    # Slow, consistent degrading trend with high R², low volatility, no discontinuity
    if (profile.drift_direction == "degrading" and
            AGING_DRIFT_MAX_SLOPE <= profile.drift_rate < 0 and
            r_squared > 0.4 and
            profile.volatility < MEDICAL_EVENT_VOLATILITY and
            discontinuity < 0.15):
        profile.classification = "natural_aging"
        profile.confidence = min(0.95, 0.5 + r_squared * 0.4)
        return profile

    # Slower aging fallback — must still have no discontinuity
    if (profile.drift_direction == "degrading" and
            profile.drift_rate < 0 and
            profile.volatility < 0.1 and
            discontinuity < 0.15):
        profile.classification = "natural_aging"
        profile.confidence = min(0.85, 0.4 + r_squared * 0.3)
        return profile

    # Rule 4: NORMAL
    profile.classification = "normal"
    profile.confidence = min(0.95, 0.6 + (1 - profile.volatility) * 0.3)
    return profile


def _compute_short_trend(scores: np.ndarray) -> float:
    """Slope of the most recent SHORT_WINDOW scores."""
    if len(scores) < 3:
        return 0.0
    recent = scores[-SHORT_WINDOW:]
    x = np.arange(len(recent), dtype=np.float64)
    slope, _, _, _, _ = stats.linregress(x, recent)
    return float(slope)


def _compute_long_trend(scores: np.ndarray) -> float:
    """Slope over all available scores."""
    if len(scores) < 3:
        return 0.0
    x = np.arange(len(scores), dtype=np.float64)
    slope, _, _, _, _ = stats.linregress(x, scores)
    return float(slope)


def _detect_recent_drop(scores: np.ndarray) -> float:
    """
    Detect if there was a sudden drop in the most recent scores
    compared to the historical average.
    Returns: magnitude of the drop (0 = no drop).
    """
    # Need at least 3 historical samples BEYOND the recent window —
    # with len == SHORT_WINDOW the historical slice is empty and
    # np.mean(empty) silently produces NaN that poisons the profile.
    if len(scores) < SHORT_WINDOW + 3:
        return 0.0

    historical_mean = float(np.mean(scores[:-SHORT_WINDOW]))
    recent_mean = float(np.mean(scores[-SHORT_WINDOW:]))

    drop = historical_mean - recent_mean
    return max(0.0, drop)  # only positive drops count


def _detect_discontinuity(scores: np.ndarray) -> float:
    """
    Change-point detection: scan for the largest step-down in the series.
    For each possible split point, compute the mean on each side and
    return the maximum difference (mean_before - mean_after).

    This catches medical events where scores drop and stay low,
    regardless of where in the timeline the event occurred.
    """
    if len(scores) < 6:
        return 0.0

    max_drop = 0.0
    # Try each possible split point (leaving at least 3 samples on each side)
    for split in range(3, len(scores) - 2):
        before = scores[:split]
        after = scores[split:]
        drop = float(np.mean(before) - np.mean(after))
        if drop > max_drop:
            max_drop = drop

    return max_drop


def generate_explanation(
    profile: DriftProfile,
    raw_score: float,
    decision: str,
) -> str:
    """
    Generate a plain-language explanation of the verification decision.
    This is what makes the system explainable.
    """
    parts = []

    # Score context
    if raw_score >= 0.8:
        parts.append(f"The signature has a high similarity score of {raw_score:.2f}.")
    elif raw_score >= 0.6:
        parts.append(f"The signature has a moderate similarity score of {raw_score:.2f}.")
    else:
        parts.append(f"The signature has a low similarity score of {raw_score:.2f}.")

    # Tenure context
    years = profile.tenure_months // 12
    months = profile.tenure_months % 12
    if years > 0:
        tenure_str = f"{years} year{'s' if years != 1 else ''}"
        if months > 0:
            tenure_str += f" and {months} month{'s' if months != 1 else ''}"
    else:
        tenure_str = f"{months} month{'s' if months != 1 else ''}"
    parts.append(f"The user has been enrolled for {tenure_str}.")

    # Trust buffer
    if profile.trust_buffer > 0:
        user_base = profile.adjusted_threshold + profile.trust_buffer
        parts.append(
            f"A trust buffer of {profile.trust_buffer:.2f} has been applied, "
            f"lowering the threshold from {user_base:.2f} to {profile.adjusted_threshold:.2f}."
        )

    # Drift classification
    if profile.classification == "natural_aging":
        parts.append(
            "The signature shows a pattern consistent with natural aging — "
            "slow, steady, directional drift with low volatility. "
            "This is expected and does not indicate fraud."
        )
        if profile.aging_allowance > 0:
            parts.append(
                f"Because the change matches this user's established aging "
                f"trend, the acceptance threshold was relaxed by a further "
                f"{profile.aging_allowance:.2f} to {profile.adjusted_threshold - profile.aging_allowance:.2f}, "
                f"so they are judged against where their signature is "
                f"expected to be today rather than at enrollment."
            )
    elif profile.classification == "medical_event":
        parts.append(
            "A sudden, significant change in signature quality was detected, "
            "which may indicate a medical event (stroke, injury, neurological change). "
            "This does not appear to be fraud but requires human review."
        )
    elif profile.classification == "forgery_attempt":
        parts.append(
            "The signature scores show a suspicious pattern of rapid improvement over "
            "a short period. Combined with abnormally smooth strokes (low tremor), "
            "this is consistent with a practiced forgery attempt."
        )
    else:
        parts.append("The signature drift pattern appears normal and stable.")

    # Decision
    if decision == "approved":
        parts.append("DECISION: Approved.")
    elif decision == "escalated":
        parts.append("DECISION: Escalated for human review.")
    else:
        parts.append("DECISION: Rejected.")

    return " ".join(parts)