from typing import List, Optional, Dict, Any
from datetime import datetime
import io

from config import MAX_SPRITE_BYTES, ALLOWED_SPRITE_FORMATS
from services.supabase import get_supabase


# Supabase bucket names
PENDING_BUCKET = "sprites-pending"
APPROVED_BUCKET = "sprites-approved"


class SpriteStorage:
    """Handle sprite file storage with pending/approved workflow using Supabase Storage."""
    
    def __init__(self):
        self.client = get_supabase()
        if not self.client:
            raise RuntimeError("Supabase not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY")
    
    def _validate_image(self, content_type: str, file_size: int) -> None:
        """Validate image upload."""
        if content_type not in ALLOWED_SPRITE_FORMATS:
            raise ValueError(
                f"Invalid format. Allowed: {', '.join(ALLOWED_SPRITE_FORMATS)}"
            )
        
        if file_size > MAX_SPRITE_BYTES:
            raise ValueError(
                f"File too large ({file_size / 1024 / 1024:.1f}MB). Max: {MAX_SPRITE_BYTES / 1024 / 1024}MB"
            )
    
    def _get_extension(self, content_type: str) -> str:
        """Get file extension from content type."""
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
        }
        return ext_map.get(content_type, ".png")
    
    def save_pending(
        self,
        user_id: str,
        image_data: bytes,
        content_type: str,
        original_filename: Optional[str] = None,
    ) -> str:
        """Save kid's upload to Supabase pending bucket.
        
        Returns:
            Filename of saved pending sprite
        """
        self._validate_image(content_type, len(image_data))
        
        # Build storage path: user_id/filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = self._get_extension(content_type)
        
        if original_filename:
            clean_name = "".join(c for c in original_filename if c.isalnum() or c in "._- ")
            clean_name = clean_name[:50]
            filename = f"{timestamp}_{clean_name}"
            if not filename.endswith(ext):
                filename += ext
        else:
            filename = f"{timestamp}_sprite{ext}"
        
        storage_path = f"{user_id}/{filename}"
        
        # Upload to Supabase Storage
        self.client.storage.from_(PENDING_BUCKET).upload(
            storage_path,
            image_data,
            file_options={"content-type": content_type}
        )
        
        return filename
    
    def list_pending(self, user_id: Optional[str] = None) -> Dict[str, List[str]]:
        """List all pending uploads.
        
        Args:
            user_id: If provided, only show this user's pending sprites
        
        Returns:
            Dict mapping user_id to list of filenames
        """
        result = {}
        
        if user_id:
            # List files in specific user folder
            files = self.client.storage.from_(PENDING_BUCKET).list(user_id)
            result[user_id] = [f["name"] for f in files if f.get("name")]
        else:
            # List all top-level folders (user IDs)
            folders = self.client.storage.from_(PENDING_BUCKET).list()
            for folder in folders:
                folder_name = folder.get("name")
                if folder_name and folder.get("id"):  # is a folder
                    files = self.client.storage.from_(PENDING_BUCKET).list(folder_name)
                    result[folder_name] = [f["name"] for f in files if f.get("name")]
        
        return result
    
    def approve_sprite(
        self,
        user_id: str,
        image_data: bytes,
        content_type: str,
        sprite_name: str,
    ) -> str:
        """Save admin's approved/redrawn sprite to Supabase approved bucket.
        
        Args:
            user_id: User to approve sprite for
            image_data: Final sprite image
            content_type: Image MIME type
            sprite_name: Desired sprite filename (without extension)
        
        Returns:
            Filename of approved sprite
        """
        self._validate_image(content_type, len(image_data))
        
        # Clean sprite name
        clean_name = "".join(c for c in sprite_name if c.isalnum() or c in "_-")
        clean_name = clean_name[:50]
        
        ext = self._get_extension(content_type)
        filename = f"{clean_name}{ext}"
        
        storage_path = f"{user_id}/{filename}"
        
        # Upload to approved bucket (upsert if exists)
        self.client.storage.from_(APPROVED_BUCKET).upload(
            storage_path,
            image_data,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        return filename
    
    def list_approved(self, user_id: str) -> List[str]:
        """List user's approved sprites."""
        try:
            files = self.client.storage.from_(APPROVED_BUCKET).list(user_id)
            return [f["name"] for f in files if f.get("name")]
        except Exception:
            return []
    
    def get_sprite_url(
        self,
        user_id: str,
        filename: str,
        pending: bool = False,
    ) -> Optional[str]:
        """Get public URL for sprite.
        
        Returns signed URL (valid for 1 hour) or None if not found.
        """
        bucket = PENDING_BUCKET if pending else APPROVED_BUCKET
        storage_path = f"{user_id}/{filename}"
        
        try:
            # Create signed URL (1 hour expiry)
            url_data = self.client.storage.from_(bucket).create_signed_url(
                storage_path,
                expires_in=3600  # 1 hour
            )
            return url_data.get("signedURL")
        except Exception as e:
            print(f"Error getting sprite URL: {e}")
            return None
    
    def get_sprite_bytes(
        self,
        user_id: str,
        filename: str,
        pending: bool = False,
    ) -> Optional[bytes]:
        """Download sprite bytes from Supabase Storage."""
        bucket = PENDING_BUCKET if pending else APPROVED_BUCKET
        storage_path = f"{user_id}/{filename}"
        
        try:
            data = self.client.storage.from_(bucket).download(storage_path)
            return data
        except Exception as e:
            print(f"Error downloading sprite: {e}")
            return None
    
    def delete_pending(self, user_id: str, filename: str) -> bool:
        """Delete a pending sprite after review."""
        storage_path = f"{user_id}/{filename}"
        
        try:
            self.client.storage.from_(PENDING_BUCKET).remove([storage_path])
            return True
        except Exception as e:
            print(f"Error deleting pending sprite: {e}")
            return False


# Singleton instance
_storage: Optional[SpriteStorage] = None


def get_storage() -> SpriteStorage:
    """Get sprite storage singleton."""
    global _storage
    if _storage is None:
        _storage = SpriteStorage()
    return _storage