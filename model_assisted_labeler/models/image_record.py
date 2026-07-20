from dataclasses import dataclass, field
from pathlib import Path

from model_assisted_labeler.models.bounding_box import BoundingBox


@dataclass
class ImageRecord:
    """
    Represents one source image in an annotation session.

    Source images are read-only. ``label_path`` always points into the
    program-owned session ``Annotations`` directory.
    """

    image_path: Path
    label_path: Path

    width: int
    height: int

    annotations: list[BoundingBox] = field(default_factory=list)

    is_dirty: bool = False
    predictions_loaded: bool = False
    annotations_loaded: bool = False
    in_annotation_pool: bool = False

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        self.label_path = Path(self.label_path)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be greater than zero.")

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    @property
    def has_annotations(self) -> bool:
        return bool(self.annotations)

    def add_annotation(self, box: BoundingBox) -> None:
        box.normalize_coordinates()
        box.clamp(self.width, self.height)

        if not box.is_valid():
            raise ValueError("Cannot add an invalid bounding box.")

        self.annotations.append(box)
        self.annotations_loaded = True
        self.is_dirty = True

    def update_annotation(
        self,
        index: int,
        updated_box: BoundingBox,
    ) -> None:
        self._validate_annotation_index(index)

        updated_box.normalize_coordinates()
        updated_box.clamp(self.width, self.height)

        if not updated_box.is_valid():
            raise ValueError("Cannot store an invalid bounding box.")

        self.annotations[index] = updated_box
        self.annotations_loaded = True
        self.is_dirty = True

    def remove_annotation(self, index: int) -> BoundingBox:
        self._validate_annotation_index(index)

        removed_box = self.annotations.pop(index)
        self.annotations_loaded = True
        self.is_dirty = True

        return removed_box

    def clear_annotations(self) -> None:
        if self.annotations:
            self.annotations.clear()
            self.annotations_loaded = True
            self.is_dirty = True

    def replace_annotations(
        self,
        annotations: list[BoundingBox],
    ) -> None:
        validated_annotations = self._validated_annotations(
            annotations
        )

        self.annotations = validated_annotations
        self.annotations_loaded = True
        self.is_dirty = True

    def load_annotations(
        self,
        annotations: list[BoundingBox],
        in_annotation_pool: bool,
    ) -> None:
        """
        Install annotations loaded from session storage without marking
        the record as modified.
        """
        self.annotations = self._validated_annotations(annotations)
        self.annotations_loaded = True
        self.in_annotation_pool = bool(in_annotation_pool)
        self.is_dirty = False
        self.predictions_loaded = False

    def unload_annotations(self) -> bool:
        """
        Release clean cached annotations.

        Dirty records are intentionally retained so unsaved work is not
        lost while the user navigates.
        """
        if self.is_dirty:
            contains_user_work = any(
                box.source != "model"
                for box in self.annotations
            )

            if contains_user_work or self.in_annotation_pool:
                return False

            # Untouched model proposals can be generated again and do
            # not need to remain cached indefinitely.
            self.is_dirty = False

        self.annotations = []
        self.annotations_loaded = False
        self.predictions_loaded = False
        return True

    def mark_saved(self) -> None:
        self.is_dirty = False
        self.annotations_loaded = True
        self.in_annotation_pool = True

    def mark_removed_from_pool(self) -> None:
        self.annotations = []
        self.annotations_loaded = True
        self.in_annotation_pool = False
        self.is_dirty = False
        self.predictions_loaded = False

    def mark_predictions_loaded(self) -> None:
        self.predictions_loaded = True

    def _validated_annotations(
        self,
        annotations: list[BoundingBox],
    ) -> list[BoundingBox]:
        validated_annotations: list[BoundingBox] = []

        for box in annotations:
            box.normalize_coordinates()
            box.clamp(self.width, self.height)

            if box.is_valid():
                validated_annotations.append(box)

        return validated_annotations

    def _validate_annotation_index(self, index: int) -> None:
        if index < 0 or index >= len(self.annotations):
            raise IndexError(
                f"Annotation index {index} is out of range."
            )
