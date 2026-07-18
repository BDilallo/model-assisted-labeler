from dataclasses import dataclass, field
from pathlib import Path

from model_assisted_labeler.models.bounding_box import BoundingBox


@dataclass
class ImageRecord:
    """
    Represents one image in the labeling project.

    Stores the image location, its dimensions, and the current
    collection of bounding-box annotations.
    """

    image_path: Path
    label_path: Path

    width: int
    height: int

    annotations: list[BoundingBox] = field(default_factory=list)

    is_dirty: bool = False
    predictions_loaded: bool = False

    def __post_init__(self) -> None:
        """
        Validate and normalize values immediately after the
        dataclass initializer runs.
        """
        self.image_path = Path(self.image_path)
        self.label_path = Path(self.label_path)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be greater than zero.")

    @property
    def filename(self) -> str:
        """Return the image filename, including its extension."""
        return self.image_path.name

    @property
    def annotation_count(self) -> int:
        """Return the number of annotations assigned to the image."""
        return len(self.annotations)

    def add_annotation(self, box: BoundingBox) -> None:
        """
        Add a bounding box to the image.

        The box is normalized and kept inside the image boundaries
        before it is accepted.
        """
        box.normalize_coordinates()
        box.clamp(self.width, self.height)

        if not box.is_valid():
            raise ValueError("Cannot add an invalid bounding box.")

        self.annotations.append(box)
        self.is_dirty = True

    def update_annotation(
        self,
        index: int,
        updated_box: BoundingBox,
    ) -> None:
        """
        Replace an existing annotation with an updated bounding box.
        """
        self._validate_annotation_index(index)

        updated_box.normalize_coordinates()
        updated_box.clamp(self.width, self.height)

        if not updated_box.is_valid():
            raise ValueError("Cannot store an invalid bounding box.")

        self.annotations[index] = updated_box
        self.is_dirty = True

    def remove_annotation(self, index: int) -> BoundingBox:
        """
        Remove and return an annotation using its list index.
        """
        self._validate_annotation_index(index)

        removed_box = self.annotations.pop(index)
        self.is_dirty = True

        return removed_box

    def clear_annotations(self) -> None:
        """Remove every annotation from the image."""
        if self.annotations:
            self.annotations.clear()
            self.is_dirty = True

    def replace_annotations(
        self,
        annotations: list[BoundingBox],
    ) -> None:
        """
        Replace all current annotations with a new collection.
        """
        validated_annotations: list[BoundingBox] = []

        for box in annotations:
            box.normalize_coordinates()
            box.clamp(self.width, self.height)

            if not box.is_valid():
                continue

            validated_annotations.append(box)

        self.annotations = validated_annotations
        self.is_dirty = True

    def mark_saved(self) -> None:
        """Record that the current annotation state has been saved."""
        self.is_dirty = False

    def mark_predictions_loaded(self) -> None:
        """Record that model inference has run for this image."""
        self.predictions_loaded = True

    def _validate_annotation_index(self, index: int) -> None:
        """
        Raise an error when an annotation index does not exist.
        """
        if index < 0 or index >= len(self.annotations):
            raise IndexError(
                f"Annotation index {index} is out of range."
            )