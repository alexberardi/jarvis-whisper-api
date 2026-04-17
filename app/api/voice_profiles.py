"""Voice profile enrollment API endpoints.

Allows uploading, deleting, and listing voice profiles for speaker
identification. Profiles are stored on the filesystem and used by
the speaker recognition pipeline during transcription.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from jarvis_auth_client.models import AppAuthResult

from app.deps import verify_app_auth
from app.utils import PROFILE_DIR, hash_user_id, invalidate_household_cache, recognize_speaker

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/voice-profiles", tags=["voice-profiles"])


@router.post("/enroll")
async def enroll_voice_profile(
    user_id: int,
    household_id: str,
    file: UploadFile = File(...),
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Upload a WAV voice sample and save as a speaker profile.

    The file is saved as voice_profiles/{household_id}/{hash(user_id)}.wav.
    Any existing profile for the same user is overwritten.
    """
    household_dir = PROFILE_DIR / household_id
    household_dir.mkdir(parents=True, exist_ok=True)

    filename = hash_user_id(user_id) + ".wav"
    filepath = household_dir / filename

    # Write to temp file first, then move (atomic on same filesystem)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=str(household_dir), delete=False
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        os.replace(tmp_path, str(filepath))
    except OSError:
        # Fallback for cross-device rename
        shutil.move(tmp_path, str(filepath))

    # Invalidate cache so next transcription picks up the new profile
    invalidate_household_cache(household_id)

    logger.info(f"Enrolled voice profile for user_id={user_id} in household={household_id}")
    return {
        "status": "enrolled",
        "user_id": user_id,
        "household_id": household_id,
    }


@router.delete("/{user_id}")
async def delete_voice_profile(
    user_id: int,
    household_id: str,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Remove a voice profile for a user."""
    household_dir = PROFILE_DIR / household_id
    filename = hash_user_id(user_id) + ".wav"
    filepath = household_dir / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found")

    filepath.unlink()
    invalidate_household_cache(household_id)

    logger.info(f"Deleted voice profile for user_id={user_id} in household={household_id}")
    return {"status": "deleted", "user_id": user_id, "household_id": household_id}


@router.get("")
async def list_voice_profiles(
    household_id: str,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """List enrolled voice profiles for a household.

    Returns file hashes (not user IDs) since the mapping is one-way.
    """
    household_dir = PROFILE_DIR / household_id

    if not household_dir.exists():
        return {"household_id": household_id, "profiles": []}

    profiles = []
    for wav_file in household_dir.glob("*.wav"):
        profiles.append({
            "filename": wav_file.name,
            "size_bytes": wav_file.stat().st_size,
        })

    return {"household_id": household_id, "profiles": profiles}


@router.get("/check")
async def check_voice_profile(
    user_id: int,
    household_id: str,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Check whether a voice profile exists for a specific user."""
    filepath = PROFILE_DIR / household_id / (hash_user_id(user_id) + ".wav")
    return {"exists": filepath.exists(), "user_id": user_id}


@router.post("/verify")
async def verify_voice_profile(
    user_id: int,
    household_id: str,
    file: UploadFile = File(...),
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Test whether an audio sample matches a user's enrolled voice profile.

    Runs resemblyzer speaker recognition only (no whisper transcription),
    so this is fast (~150ms). Used by the mobile app's enrollment wizard
    to let the user confirm their profile works before walking away.
    """
    # Check profile exists first
    profile_path = PROFILE_DIR / household_id / (hash_user_id(user_id) + ".wav")
    if not profile_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No voice profile enrolled for user {user_id}",
        )

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=tempfile.gettempdir(), delete=False
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = recognize_speaker(
            audio_path=tmp_path,
            household_id=household_id,
            member_ids=[user_id],
            threshold=0.65,  # Lower than default 0.75 for 1:1 targeted match
        )
    finally:
        os.unlink(tmp_path)

    matched = result.user_id is not None
    logger.info(
        f"Voice profile verify: user_id={user_id} matched={matched} "
        f"confidence={result.confidence:.3f}"
    )

    return {
        "matched": matched,
        "confidence": round(result.confidence, 4),
        "user_id": user_id,
    }
