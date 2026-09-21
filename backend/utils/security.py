import os
import time
import uuid
import logging
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024 # 15 MB limit

def is_allowed_file(filename):
    """Checks if file extension is allowed."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def generate_safe_filename(original_filename):
    """Generates a secure, sanitized unique filename."""
    basename = secure_filename(original_filename)
    if not basename:
        basename = "audio_upload"
    
    name, ext = os.path.splitext(basename)
    if not ext:
        ext = ".wav"
    
    unique_id = uuid.uuid4().hex[:8]
    return f"{name}_{unique_id}{ext}"

def validate_file_size(file_obj):
    """Verifies file size does not exceed MAX_FILE_SIZE_BYTES."""
    file_obj.seek(0, os.SEEK_END)
    size = file_obj.tell()
    file_obj.seek(0)
    if size > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({round(size / (1024 * 1024), 2)} MB) exceeds 15 MB maximum limit."
    return True, None

def cleanup_old_uploads(upload_folder, max_age_seconds=3600):
    """Deletes uploaded temporary files older than max_age_seconds (default 1 hour)."""
    now = time.time()
    try:
        for filename in os.listdir(upload_folder):
            if filename == ".gitkeep":
                continue
            filepath = os.path.join(upload_folder, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up stale upload: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing {filename}: {e}")
    except Exception as e:
        logger.error(f"Error during upload directory cleanup: {e}")
