"""Thought signature cache for Gemini API."""

import json
import time
from pathlib import Path
from typing import Dict, Optional


class ThoughtSignatureCache:
    """Cache for storing and retrieving Gemini thought signatures."""

    def __init__(self, cache_dir: str = "./tmp"):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "thought_signatures.json"
        self._load_cache()

    def _load_cache(self):
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('signatures', {})
                    self.metadata = data.get('metadata', {})
            except Exception:
                self.cache = {}
                self.metadata = {}
        else:
            self.cache = {}
            self.metadata = {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            data = {
                'signatures': self.cache,
                'metadata': self.metadata,
                'last_updated': time.time()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def store_signature(self, tool_call_id: str, thought_signature: str, request_id: str = ""):
        """
        Store thought_signature for a tool call.

        Args:
            tool_call_id: Unique ID of the tool call
            thought_signature: The signature to store
            request_id: Optional request ID for tracking
        """
        self.cache[tool_call_id] = thought_signature
        self.metadata[tool_call_id] = {
            'request_id': request_id,
            'timestamp': time.time()
        }
        self._save_cache()

    def get_signature(self, tool_call_id: str) -> Optional[str]:
        """
        Get stored thought_signature for a tool call.

        Args:
            tool_call_id: Unique ID of the tool call

        Returns:
            The stored signature or None if not found
        """
        return self.cache.get(tool_call_id)

    def cleanup_old_entries(self, max_age_seconds: int = 3600, max_entries: int = 1000):
        """
        Remove old cache entries.

        Args:
            max_age_seconds: Maximum age of entries in seconds (default: 1 hour)
            max_entries: Maximum number of entries to keep
        """
        current_time = time.time()

        # Remove entries older than max_age_seconds
        old_keys = []
        for key, meta in self.metadata.items():
            if current_time - meta.get('timestamp', 0) > max_age_seconds:
                old_keys.append(key)

        for key in old_keys:
            self.cache.pop(key, None)
            self.metadata.pop(key, None)

        # If still too many entries, keep only the most recent ones
        if len(self.cache) > max_entries:
            # Sort by timestamp, keep newest
            sorted_items = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get('timestamp', 0),
                reverse=True
            )

            keys_to_keep = set(k for k, _ in sorted_items[:max_entries])

            self.cache = {k: v for k, v in self.cache.items() if k in keys_to_keep}
            self.metadata = {k: v for k, v in self.metadata.items() if k in keys_to_keep}

        self._save_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists()
        }
