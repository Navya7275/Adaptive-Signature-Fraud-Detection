"""
Verification API routes — the core endpoint.
"""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.verification import verify_signature

router = APIRouter(prefix="/api/verify", tags=["Verification"])


@router.post("/")
async def verify(
    user_id: str = Form(...),
    signature: UploadFile = File(...),
):
    """
    Verify a signature against a user's adaptive profile.

    Upload a signature image and provide the user_id.
    Returns a full verification result with:
      - decision (approved/escalated/rejected)
      - similarity scores
      - drift classification
      - plain-language explanation
    """
    # Save uploaded file
    suffix = os.path.splitext(signature.filename)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await signature.read()
        tmp.write(content)
        tmp.close()

        result = verify_signature(user_id, tmp.name)

        if "error" in result:
            raise HTTPException(400, result["error"])

        return result

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.get("/logs/{user_id}")
async def get_verification_logs(user_id: str, limit: int = 20):
    """Get recent verification logs for a user."""
    from app.database.db import get_db, rows_to_dicts

    with get_db() as db:
        logs = rows_to_dicts(db.execute(
            """SELECT * FROM verification_logs
               WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall())

    return logs


@router.get("/alerts")
async def get_all_alerts(resolved: bool = False):
    """Get all active alerts across all users."""
    from app.database.db import get_db, rows_to_dicts

    with get_db() as db:
        alerts = rows_to_dicts(db.execute(
            """SELECT a.*, u.name as user_name
               FROM alerts a JOIN users u ON a.user_id = u.id
               WHERE a.resolved = ?
               ORDER BY a.created_at DESC""",
            (int(resolved),)
        ).fetchall())

    return alerts


@router.put("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Mark an alert as resolved."""
    from app.database.db import get_db

    with get_db() as db:
        cur = db.execute("UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, f"Alert {alert_id} not found")
    return {"message": "Alert resolved", "alert_id": alert_id}