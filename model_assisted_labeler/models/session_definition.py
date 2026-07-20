from dataclasses import dataclass, field
from pathlib import Path

from model_assisted_labeler.models.annotation_session import ClassDefinition


@dataclass
class SessionDefinition:
    """
    Persistent configuration for one annotation session.

    The definition points at the read-only source image directory and
    the program-owned session directory. Model paths are stored as a
    list so support for multiple models can be added without changing
    the on-disk format later.
    """

    name: str
    session_directory: Path
    image_directory: Path
    model_paths: list[Path] = field(default_factory=list)
    classes: list[ClassDefinition] = field(default_factory=list)
    last_image_loaded: str | None = None
    total_images_annotated: int = 0
    next_class_id: int = 0

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.session_directory = Path(self.session_directory)
        self.image_directory = Path(self.image_directory)
        self.model_paths = [Path(path) for path in self.model_paths]

        if not self.name:
            raise ValueError("Session name cannot be empty.")

        if self.total_images_annotated < 0:
            raise ValueError(
                "Total annotated image count cannot be negative."
            )

        if self.next_class_id < 0:
            raise ValueError("Next class ID cannot be negative.")

    @property
    def primary_model_path(self) -> Path | None:
        """Return the first configured model path, if one exists."""
        if not self.model_paths:
            return None

        return self.model_paths[0]
