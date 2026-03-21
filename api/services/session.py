"""Session tracking service for parent progress monitoring."""

from typing import Optional, Dict, Any, List

from services.supabase import get_supabase


def start_session(user_id: str) -> Optional[str]:
    """Start a new practice session.

    Args:
        user_id: Supabase auth user ID.

    Returns:
        session_id (UUID string) on success, None if Supabase is unavailable.
    """
    client = get_supabase()
    if not client:
        return None

    result = (
        client.table("session_logs")
        .insert({"user_id": user_id})
        .execute()
    )
    rows = result.data
    if rows:
        return rows[0]["id"]
    return None


def end_session(session_id: str, message_count: int) -> bool:
    """End a session, computing duration server-side.

    Sets ended_at = now() and duration_seconds = ended_at - started_at.

    Args:
        session_id: UUID of the session to close.
        message_count: Total messages exchanged during this session.

    Returns:
        True if the row was updated, False otherwise.
    """
    client = get_supabase()
    if not client:
        return False

    # Two-step: set ended_at first, then compute duration from the stored started_at.
    # We fetch started_at so we can compute duration in Python
    # (Supabase REST doesn't support column-reference arithmetic in PATCH).
    fetch = (
        client.table("session_logs")
        .select("started_at")
        .eq("id", session_id)
        .is_("ended_at", "null")
        .execute()
    )
    if not fetch.data:
        return False

    from datetime import datetime, timezone

    started_at_str = fetch.data[0]["started_at"]
    # Parse ISO timestamp (Supabase returns ISO 8601 with tz)
    started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    duration = int((now - started_at).total_seconds())

    result = (
        client.table("session_logs")
        .update({
            "ended_at": now.isoformat(),
            "duration_seconds": max(duration, 0),
            "message_count": max(message_count, 0),
        })
        .eq("id", session_id)
        .execute()
    )
    return bool(result.data)


def get_progress(user_id: str) -> Dict[str, Any]:
    """Aggregate progress stats for a user.

    Returns:
        Dict with total_sessions, total_duration_seconds, total_messages,
        last_session_at, and recent_sessions (last 10 completed).
    """
    client = get_supabase()
    if not client:
        return _empty_progress()

    # Fetch last 10 completed sessions (ended_at IS NOT NULL), newest first.
    result = (
        client.table("session_logs")
        .select("id, started_at, ended_at, duration_seconds, message_count")
        .eq("user_id", user_id)
        .not_.is_("ended_at", "null")
        .order("started_at", desc=True)
        .limit(10)
        .execute()
    )
    recent: List[Dict[str, Any]] = result.data or []

    # Fetch aggregates across ALL completed sessions (not just last 10).
    all_result = (
        client.table("session_logs")
        .select("duration_seconds, message_count")
        .eq("user_id", user_id)
        .not_.is_("ended_at", "null")
        .execute()
    )
    all_rows = all_result.data or []

    total_sessions = len(all_rows)
    total_duration = sum(r.get("duration_seconds") or 0 for r in all_rows)
    total_messages = sum(r.get("message_count") or 0 for r in all_rows)
    last_session_at = recent[0]["started_at"] if recent else None

    return {
        "total_sessions": total_sessions,
        "total_duration_seconds": total_duration,
        "total_messages": total_messages,
        "last_session_at": last_session_at,
        "recent_sessions": recent,
    }


def _empty_progress() -> Dict[str, Any]:
    return {
        "total_sessions": 0,
        "total_duration_seconds": 0,
        "total_messages": 0,
        "last_session_at": None,
        "recent_sessions": [],
    }