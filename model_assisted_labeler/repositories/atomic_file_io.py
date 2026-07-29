from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to ``path`` so readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
