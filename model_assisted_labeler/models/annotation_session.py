from dataclasses import dataclass, field
from pathlib import Path

from model_assisted_labeler.models.image_record import ImageRecord


@dataclass(frozen=True)
class ClassDefinition:
    """Represents one session annotation class."""

    class_id: int
    name: str

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("Class ID cannot be negative.")

        cleaned_name = self.name.strip()

        if not cleaned_name:
            raise ValueError("Class name cannot be empty.")

        object.__setattr__(self, "name", cleaned_name)


@dataclass
class AnnotationSession:
    """
    Active in-memory annotation session.

    Image metadata is kept for the full source directory, while box
    annotations are loaded lazily by the controller.
    """

    name: str
    session_directory: Path
    image_directory: Path
    label_directory: Path

    classes: list[ClassDefinition] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)

    current_index: int = 0

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.session_directory = Path(self.session_directory)
        self.image_directory = Path(self.image_directory)
        self.label_directory = Path(self.label_directory)

        if not self.name:
            raise ValueError("Session name cannot be empty.")

        self._validate_classes()
        self._validate_images()
        self._normalize_current_index()

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def current_image(self) -> ImageRecord | None:
        if not self.images:
            return None

        return self.images[self.current_index]

    @property
    def current_position(self) -> int:
        if not self.images:
            return 0

        return self.current_index + 1

    @property
    def is_first_image(self) -> bool:
        return bool(self.images) and self.current_index == 0

    @property
    def is_last_image(self) -> bool:
        return (
            bool(self.images)
            and self.current_index == len(self.images) - 1
        )

    def next_image(self) -> ImageRecord | None:
        if not self.images:
            return None

        if not self.is_last_image:
            self.current_index += 1

        return self.current_image

    def previous_image(self) -> ImageRecord | None:
        if not self.images:
            return None

        if not self.is_first_image:
            self.current_index -= 1

        return self.current_image

    def go_to_image(self, index: int) -> ImageRecord:
        self._validate_image_index(index)
        self.current_index = index
        return self.images[index]

    def find_image_index(self, filename: str | None) -> int | None:
        if not filename:
            return None

        target = filename.casefold()

        for index, image_record in enumerate(self.images):
            if image_record.filename.casefold() == target:
                return index

        return None

    def get_class(
        self,
        class_id: int,
    ) -> ClassDefinition | None:
        for class_definition in self.classes:
            if class_definition.class_id == class_id:
                return class_definition

        return None

    def get_class_by_name(
        self,
        class_name: str,
    ) -> ClassDefinition | None:
        normalized_name = class_name.strip().casefold()

        for class_definition in self.classes:
            if class_definition.name.casefold() == normalized_name:
                return class_definition

        return None

    def get_class_name(self, class_id: int) -> str | None:
        class_definition = self.get_class(class_id)

        if class_definition is None:
            return None

        return class_definition.name

    def add_class(self, class_definition: ClassDefinition) -> None:
        if self.get_class(class_definition.class_id) is not None:
            raise ValueError(
                f"Class ID {class_definition.class_id} already exists."
            )

        if self.get_class_by_name(class_definition.name) is not None:
            raise ValueError(
                f"Class '{class_definition.name}' already exists."
            )

        self.classes.append(class_definition)
        self.classes.sort(key=lambda item: item.class_id)

    def remove_class(self, class_id: int) -> ClassDefinition:
        for index, class_definition in enumerate(self.classes):
            if class_definition.class_id == class_id:
                return self.classes.pop(index)

        raise ValueError(f"Class ID {class_id} is not defined.")

    def has_unsaved_changes(self) -> bool:
        return any(image.is_dirty for image in self.images)

    def dirty_images(self) -> list[ImageRecord]:
        return [image for image in self.images if image.is_dirty]

    def _validate_image_index(self, index: int) -> None:
        if index < 0 or index >= len(self.images):
            raise IndexError(f"Image index {index} is out of range.")

    def _normalize_current_index(self) -> None:
        if not self.images:
            self.current_index = 0
        else:
            self.current_index = min(
                max(self.current_index, 0),
                len(self.images) - 1,
            )

    def _validate_classes(self) -> None:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()

        for class_definition in self.classes:
            if not isinstance(class_definition, ClassDefinition):
                raise TypeError(
                    "Classes must contain only ClassDefinition objects."
                )

            if class_definition.class_id in seen_ids:
                raise ValueError(
                    f"Duplicate class ID: {class_definition.class_id}"
                )

            normalized_name = class_definition.name.casefold()

            if normalized_name in seen_names:
                raise ValueError(
                    f"Duplicate class name: {class_definition.name}"
                )

            seen_ids.add(class_definition.class_id)
            seen_names.add(normalized_name)

    def _validate_images(self) -> None:
        seen_paths: set[Path] = set()

        for image_record in self.images:
            if not isinstance(image_record, ImageRecord):
                raise TypeError(
                    "Images must contain only ImageRecord objects."
                )

            if image_record.image_path in seen_paths:
                raise ValueError(
                    f"Duplicate image path: {image_record.image_path}"
                )

            seen_paths.add(image_record.image_path)
