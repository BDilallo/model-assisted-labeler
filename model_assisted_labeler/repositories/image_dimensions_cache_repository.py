import json
from dataclasses import dataclass
from pathlib import Path

from model_assisted_labeler.repositories.atomic_file_io import (
    atomic_write_text,
)


@dataclass(frozen=True)
class CachedImageDimensions:
    """One cached width/height reading plus the stat fields used to
    detect that the underlying file has changed since it was read."""

    width: int
    height: int
    file_size: int
    modified_time: float


class ImageDimensionsCacheRepository:
    """
    Persists image width/height readings for a session so repeat
    session loads do not need to reopen every source image.

    Entries are keyed by filename and are only trusted when a file's
    current size and modification time still match what was recorded,
    so an image replaced in place (same name, different content) is
    detected and re-read rather than served a stale cached size.
    """

    FILENAME = "Image Dimensions.json"

    def load(
        self,
        session_directory: Path,
    ) -> dict[str, CachedImageDimensions]:
        cache_path = Path(session_directory) / self.FILENAME

        if not cache_path.is_file():
            return {}

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        if not isinstance(payload, dict):
            return {}

        entries: dict[str, CachedImageDimensions] = {}

        for filename, raw_entry in payload.items():
            entry = self._parse_entry(raw_entry)

            if entry is not None:
                entries[filename] = entry

        return entries

    def save(
        self,
        session_directory: Path,
        entries: dict[str, CachedImageDimensions],
    ) -> None:
        payload = {
            filename: {
                "width": entry.width,
                "height": entry.height,
                "size": entry.file_size,
                "mtime": entry.modified_time,
            }
            for filename, entry in entries.items()
        }

        atomic_write_text(
            Path(session_directory) / self.FILENAME,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    @staticmethod
    def _parse_entry(raw_entry: object) -> CachedImageDimensions | None:
        if not isinstance(raw_entry, dict):
            return None

        try:
            width = int(raw_entry["width"])
            height = int(raw_entry["height"])
            file_size = int(raw_entry["size"])
            modified_time = float(raw_entry["mtime"])
        except (KeyError, TypeError, ValueError):
            return None

        if width <= 0 or height <= 0 or file_size < 0:
            return None

        return CachedImageDimensions(
            width=width,
            height=height,
            file_size=file_size,
            modified_time=modified_time,
        )
