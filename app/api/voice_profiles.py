"""Voice profile enrollment API endpoints.

Allows uploading, deleting, and listing voice profiles for speaker
identification. Profiles are stored on the filesystem and used by
the speaker recognition pipeline during transcription.

Storage layout:

    voice_profiles/{household_id}/{hash(user_id)}/sample_NNN.wav

The encoder averages embeddings across all samples in a user's directory
to produce a more stable reference (multi-sample enrollment). Legacy
single-file profiles at ``{hash(user_id)}.wav`` are still supported on
read and auto-migrated to the directory layout on the next enrollment.
"""

import logging
import os
import shutil
import shlex
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from jarvis_auth_client.models import AppAuthResult

from app.deps import verify_app_auth
from app.utils import (
    PROFILE_DIR,
    hash_user_id,
    invalidate_household_cache,
    legacy_profile_path,
    migrate_legacy_profile_to_directory,
    next_sample_index,
    recognize_speaker,
    user_profile_dir,
    user_profile_paths,
)

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/voice-profiles", tags=["voice-profiles"])


def _sample_filename(index: int) -> str:
    return f"sample_{index:03d}.wav"


@router.post("/enroll")
async def enroll_voice_profile(
    user_id: int,
    household_id: str,
    file: UploadFile = File(...),
    sample_index: Optional[int] = None,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Upload a WAV voice sample and save it as a speaker profile sample.

    If ``sample_index`` is omitted, the file becomes the next sample in
    the user's directory (sample_000 if first; sample_001 if a legacy
    single-file profile is migrated, etc.). If provided, the sample at
    that index is overwritten.

    Any legacy single-file profile is migrated to ``sample_000.wav``
    before writing.
    """
    # Migrate legacy single-file profile (no-op if directory already exists)
    migrate_legacy_profile_to_directory(household_id, user_id)

    user_dir = user_profile_dir(household_id, user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    chosen_index = (
        sample_index if sample_index is not None else next_sample_index(household_id, user_id)
    )
    if chosen_index < 0 or chosen_index > 999:
        raise HTTPException(status_code=400, detail="sample_index must be in [0, 999]")

    filepath = user_dir / _sample_filename(chosen_index)

    # Write to temp file first, then move (atomic on same filesystem)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=str(user_dir), delete=False
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        os.replace(tmp_path, str(filepath))
    except OSError:
        shutil.move(tmp_path, str(filepath))

    invalidate_household_cache(household_id)

    logger.info(
        "Enrolled voice profile sample: user_id=%s household=%s index=%s",
        user_id,
        shlex.quote(household_id),
        chosen_index,
    )
    return {
        "status": "enrolled",
        "user_id": user_id,
        "household_id": household_id,
        "sample_index": chosen_index,
        "total_samples": len(user_profile_paths(household_id, user_id)),
    }


@router.delete("/{user_id}")
async def delete_voice_profile(
    user_id: int,
    household_id: str,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Remove all voice profile samples for a user.

    Cleans up both the per-user directory and any legacy single-file
    profile if present.
    """
    user_dir = user_profile_dir(household_id, user_id)
    legacy = legacy_profile_path(household_id, user_id)

    removed_any = False
    if user_dir.is_dir():
        shutil.rmtree(user_dir)
        removed_any = True
    if legacy.exists():
        legacy.unlink()
        removed_any = True

    if not removed_any:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    invalidate_household_cache(household_id)

    logger.info(f"Deleted voice profile for user_id={user_id} in household={household_id}")
    return {"status": "deleted", "user_id": user_id, "household_id": household_id}


@router.get("/{user_id}/samples")
async def list_user_samples(
    user_id: int,
    household_id: str,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """List enrollment samples for a user (multi-sample enrollment UI)."""
    paths = user_profile_paths(household_id, user_id)
    samples = []
    for p in paths:
        stem = p.stem
        try:
            idx = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            idx = 0  # legacy single-file → treat as index 0
        samples.append({
            "index": idx,
            "filename": p.name,
            "size_bytes": p.stat().st_size,
        })
    return {
        "household_id": household_id,
        "user_id": user_id,
        "samples": samples,
    }


@router.delete("/{user_id}/samples/{sample_index}")
async def delete_user_sample(
    user_id: int,
    sample_index: int,
    household_id: str,
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Delete a single enrollment sample by index (lets users redo a bad take)."""
    user_dir = user_profile_dir(household_id, user_id)
    target = user_dir / _sample_filename(sample_index)
    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sample {sample_index} not found for user {user_id}",
        )
    target.unlink()
    invalidate_household_cache(household_id)
    return {
        "status": "deleted",
        "user_id": user_id,
        "sample_index": sample_index,
        "remaining_samples": len(user_profile_paths(household_id, user_id)),
    }


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

    profiles: list[dict] = []
    # Directory-style users
    for entry in household_dir.iterdir():
        if entry.is_dir():
            sample_count = sum(1 for _ in entry.glob("sample_*.wav"))
            if sample_count:
                profiles.append({
                    "filename": entry.name,
                    "samples": sample_count,
                })
    # Legacy single-file users
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
    paths = user_profile_paths(household_id, user_id)
    return {
        "exists": bool(paths),
        "user_id": user_id,
        "sample_count": len(paths),
    }


@router.post("/verify")
async def verify_voice_profile(
    user_id: int,
    household_id: str,
    file: UploadFile = File(...),
    auth: AppAuthResult = Depends(verify_app_auth),
):
    """Test whether an audio sample matches a user's enrolled voice profile.

    Runs the speaker encoder only (no whisper transcription), so this is
    fast (~150ms). Used by the mobile app's enrollment wizard to let the
    user confirm their profile works before walking away.
    """
    if not user_profile_paths(household_id, user_id):
        raise HTTPException(
            status_code=404,
            detail=f"No voice profile enrolled for user {user_id}",
        )

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
            threshold=0.45,  # Stricter than runtime — but 1:1 targeted, so OK
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
