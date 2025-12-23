from typing import Optional
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_SECRET_KEY

_client: Optional[Client] = None


def get_supabase() -> Optional[Client]:
    """Get Supabase client (singleton)."""
    global _client
    if _client is None and SUPABASE_URL and SUPABASE_SECRET_KEY:
        _client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    return _client


def is_configured() -> bool:
    """Check if Supabase is configured."""
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY)