from dataclasses import dataclass, field
from pathlib import Path

from model_assisted_labeler.models.image_record import ImageRecord


@dataclass(frozen=True)
class ClassDefinition:
    """
    Represents one annotation class.

    Example:
        class_id = 0
        name = "face"
    """

    class_id: int
    name: str

    def __post_init__(self) -> None:
        """
        Validate and clean the class definition after initialization.
        """
        if self.class_id < 0:
            raise ValueError("Class ID cannot be negative.")

        cleaned_name = self.name.strip()

        if not cleaned_name:
            raise ValueError("Class name cannot be empty.")

        # ClassDefinition is frozen, so normal assignment is blocked.
        # object.__setattr__ allows the initial value to be cleaned here.
        object.__setattr__(self, "name", cleaned_name)


@dataclass
class AnnotationSession:
    """
    Represents the currently active annotation session.

    Tracks the loaded images, available annotation classes,
    current image position, and unsaved changes.
    """

    image_directory: Path
    label_directory: Path

    classes: list[ClassDefinition] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)

    current_index: int = 0

    def __post_init__(self) -> None:
        """
        Normalize paths and validate the initial session state.
        """
        self.image_directory = Path(self.image_directory)
        self.label_directory = Path(self.label_directory)

        self._validate_classes()
        self._validate_images()
        self._normalize_current_index()

    @property
    def image_count(self) -> int:
        """Return the total number of loaded images."""
        return len(self.images)

    @property
    def class_count(self) -> int:
        """Return the total number of annotation classes."""
        return len(self.classes)

    @property
    def has_images(self) -> bool:
        """Return True when the session contains at least one image."""
        return bool(self.images)

    @property
    def current_image(self) -> ImageRecord | None:
        """
        Return the currently selected image.

        Returns:
            The current ImageRecord, or None if the session is empty.
        """
        if not self.images:
            return None

        return self.images[self.current_index]

    @property
    def current_position(self) -> int:
        """
        Return the human-readable position of the current image.

        Internal list indexes begin at zero, while displayed positions
        begin at one. An empty session returns zero.
        """
        if not self.images:
            return 0

        return self.current_index + 1

    @property
    def is_first_image(self) -> bool:
        """Return True when positioned at the first image."""
        return bool(self.images) and self.current_index == 0

    @property
    def is_last_image(self) -> bool:
        """Return True when positioned at the final image."""
        return (
            bool(self.images)
            and self.current_index == len(self.images) - 1
        )

    def next_image(self) -> ImageRecord | None:
        """
        Move to the next image and return it.

        The session stops at the final image rather than wrapping
        back to the beginning.
        """
        if not self.images:
            return None

        if not self.is_last_image:
            self.current_index += 1

        return self.current_image

    def previous_image(self) -> ImageRecord | None:
        """
        Move to the previous image and return it.

        The session stops at the first image rather than wrapping
        to the end.
        """
        if not self.images:
            return None

        if not self.is_first_image:
            self.current_index -= 1

        return self.current_image

    def go_to_image(self, index: int) -> ImageRecord:
        """
        Select and return an image using its zero-based list index.

        Args:
            index:
                Position of the image in the session's image list.
        """
        self._validate_image_index(index)

        self.current_index = index

        return self.images[self.current_index]

    def get_class(
        self,
        class_id: int,
    ) -> ClassDefinition | None:
        """
        Find a class definition using its class ID.

        Returns:
            The matching ClassDefinition, or None if no class uses
            the supplied ID.
        """
        for class_definition in self.classes:
            if class_definition.class_id == class_id:
                return class_definition

        return None

    def get_class_name(
        self,
        class_id: int,
    ) -> str | None:
        """
        Return the class name associated with a class ID.
        """
        class_definition = self.get_class(class_id)

        if class_definition is None:
            return None

        return class_definition.name

    def has_unsaved_changes(self) -> bool:
        """
        Return True when any loaded image has unsaved changes.
        """
        return any(
            image_record.is_dirty
            for image_record in self.images
        )

    def dirty_images(self) -> list[ImageRecord]:
        """
        Return every image containing unsaved annotation changes.
        """
        return [
            image_record
            for image_record in self.images
            if image_record.is_dirty
        ]

    def add_image(
        self,
        image_record: ImageRecord,
    ) -> None:
        """
        Add an ImageRecord to the session.

        Duplicate image paths are not allowed.
        """
        if not isinstance(image_record, ImageRecord):
            raise TypeError(
                "The session can only contain ImageRecord objects."
            )

        if self._contains_image_path(image_record.image_path):
            raise ValueError(
                f"Image is already in the session: "
                f"{image_record.image_path}"
            )

        self.images.append(image_record)

        if len(self.images) == 1:
            self.current_index = 0

    def _contains_image_path(
        self,
        image_path: Path,
    ) -> bool:
        """
        Return True when an image path already exists in the session.
        """
        image_path = Path(image_path)

        return any(
            existing_image.image_path == image_path
            for existing_image in self.images
        )

    def _validate_image_index(
        self,
        index: int,
    ) -> None:
        """
        Raise an error when an image index does not exist.
        """
        if index < 0 or index >= len(self.images):
            raise IndexError(
                f"Image index {index} is out of range."
            )

    def _normalize_current_index(self) -> None:
        """
        Ensure the current index is valid for the loaded images.
        """
        if not self.images:
            self.current_index = 0
            return

        if self.current_index < 0:
            self.current_index = 0

        if self.current_index >= len(self.images):
            self.current_index = len(self.images) - 1

    def _validate_classes(self) -> None:
        """
        Ensure class definitions are valid and unique.
        """
        seen_ids: set[int] = set()
        seen_names: set[str] = set()

        for class_definition in self.classes:
            if not isinstance(class_definition, ClassDefinition):
                raise TypeError(
                    "Classes must contain only "
                    "ClassDefinition objects."
                )

            if class_definition.class_id in seen_ids:
                raise ValueError(
                    f"Duplicate class ID: "
                    f"{class_definition.class_id}"
                )

            normalized_name = class_definition.name.casefold()

            if normalized_name in seen_names:
                raise ValueError(
                    f"Duplicate class name: "
                    f"{class_definition.name}"
                )

            seen_ids.add(class_definition.class_id)
            seen_names.add(normalized_name)

    def _validate_images(self) -> None:
        """
        Ensure all initial images are valid and nonduplicate.
        """
        seen_paths: set[Path] = set()

        for image_record in self.images:
            if not isinstance(image_record, ImageRecord):
                raise TypeError(
                    "Images must contain only ImageRecord objects."
                )

            if image_record.image_path in seen_paths:
                raise ValueError(
                    f"Duplicate image path: "
                    f"{image_record.image_path}"
                )

            seen_paths.add(image_record.image_path)