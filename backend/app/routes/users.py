"""
User API routes — enrollment, listing, history.
"""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List

from app.services.user_manager import enroll_user, get_user, get_user_history, list_users

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.post("/enroll")
async def enroll(
    name: str = Form(...),
    age: int = Form(...),
    signatures: List[UploadFile] = File(...),
):
    """
    Enroll a new user with reference signatures.
    Upload 3-5 signature images for best results.
    """
    if len(signatures) < 1:
        raise HTTPException(400, "At least 1 signature image is required")
    if len(signatures) > 10:
        raise HTTPException(400, "Maximum 10 reference signatures allowed")

    # Save uploaded files to temp directory
    temp_paths = []
    try:
        for sig_file in signatures:
            suffix = os.path.splitext(sig_file.filename)[1] or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = await sig_file.read()
            tmp.write(content)
            tmp.close()
            temp_paths.append(tmp.name)

        result = enroll_user(name, age, temp_paths)

        if "error" in result:
            raise HTTPException(400, result["error"])

        return result

    finally:
        # Cleanup temp files
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


@router.get("/")
async def get_all_users():
    """List all enrolled users."""
    return list_users()


@router.get("/{user_id}")
async def get_user_detail(user_id: str):
    """Get user details with drift profile."""
    user = get_user(user_id)
    if not user:
        raise HTTPException(404, f"User {user_id} not found")
    return user


@router.get("/{user_id}/history")
async def get_history(user_id: str):
    """Get full signature and verification history."""
    return get_user_history(user_id)