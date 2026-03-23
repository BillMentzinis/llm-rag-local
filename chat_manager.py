"""
Chat persistence manager.
Saves/loads/lists/deletes chat sessions as JSON files under chats/.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from config import PATHS

CHATS_DIR = PATHS["chats"]


def _ensure_chats_dir() -> None:
    os.makedirs(CHATS_DIR, exist_ok=True)


def _safe_filename(name: str) -> str:
    """Strip characters that are illegal in Windows/Linux filenames."""
    keepchars = " .-_()[]"
    return "".join(c if (c.isalnum() or c in keepchars) else "_" for c in name).strip()


def auto_name_from_message(first_message: str, max_len: int = 40) -> str:
    """
    Derive a chat name from the first user message, truncated.
    Falls back to a timestamp string if the message is empty.
    """
    text = first_message.strip()
    if not text:
        return datetime.now().strftime("Chat_%Y%m%d_%H%M%S")
    name = text[:max_len]
    if len(text) > max_len:
        name += "..."
    return name


def save_chat(name: str, messages: List[Dict]) -> str:
    """
    Persist a chat to chats/<safe_name>.json.
    Returns the absolute path that was written.
    """
    _ensure_chats_dir()
    safe = _safe_filename(name)
    path = os.path.join(CHATS_DIR, safe + ".json")
    payload = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def list_chats() -> List[Dict]:
    """
    Return metadata for all saved chats, newest first.
    Each entry: {name, saved_at, message_count, filepath}
    """
    _ensure_chats_dir()
    entries = []
    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(CHATS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries.append({
                "name": data.get("name", fname[:-5]),
                "saved_at": data.get("saved_at", ""),
                "message_count": len(data.get("messages", [])),
                "filepath": fpath,
            })
        except (json.JSONDecodeError, OSError):
            continue
    entries.sort(key=lambda e: e["saved_at"], reverse=True)
    return entries


def load_chat(filepath: str) -> Optional[List[Dict]]:
    """
    Load messages from a saved chat file.
    Returns the messages list, or None on error.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    except (json.JSONDecodeError, OSError):
        return None


def delete_chat(filepath: str) -> bool:
    """
    Delete a saved chat file.
    Returns True on success.
    """
    try:
        os.remove(filepath)
        return True
    except OSError:
        return False
