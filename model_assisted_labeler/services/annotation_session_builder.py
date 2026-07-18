from pathlib import Path

from model_assisted_labeler.models.annotation_session import (
    AnnotationSession,
    ClassDefinition,
)
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.services.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.services.image_service import ImageService


class AnnotationSessionBuilder:
    """
    Builds an AnnotationSession from image and label directories.

    This class discovers images, reads their dimensions, loads any
    existing YOLO annotations, and creates the corresponding
    ImageRecord objects.
    """

    def __init__(
        self,
        image_service: ImageService,
        annotation_store: YoloAnnotationStore,
    ) -> None:
        """
        Store the services needed to build annotation sessions.
        """
        self.image_service = image_service
        self.annotation_store = annotation_store

    def build(
        self,
        image_directory: Path,
        label_directory: Path,
        classes: list[ClassDefinition],
        recursive: bool = False,
    ) -> AnnotationSession:
        """
        Build and return an annotation session.

        Args:
            image_directory:
                Directory containing the images to label.

            label_directory:
                Directory containing or receiving YOLO label files.

            classes:
                Available annotation classes.

            recursive:
                When True, images are discovered inside subdirectories.

        Returns:
            A fully populated AnnotationSession.
        """
        image_directory = Path(image_directory)
        label_directory = Path(label_directory)

        self._validate_classes(classes)

        image_paths = self.image_service.discover_images(
            directory=image_directory,
            recursive=recursive,
        )

        label_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_records: list[ImageRecord] = []

        for image_path in image_paths:
            image_record = self._build_image_record(
                image_path=image_path,
                image_directory=image_directory,
                label_directory=label_directory,
            )

            image_records.append(image_record)

        return AnnotationSession(
            image_directory=image_directory,
            label_directory=label_directory,
            classes=list(classes),
            images=image_records,
            current_index=0,
        )

    def _build_image_record(
        self,
        image_path: Path,
        image_directory: Path,
        label_directory: Path,
    ) -> ImageRecord:
        """
        Create one ImageRecord from an image file.
        """
        width, height = self.image_service.get_dimensions(
            image_path
        )

        label_path = self._get_label_path(
            image_path=image_path,
            image_directory=image_directory,
            label_directory=label_directory,
        )

        annotations = self.annotation_store.load(
            label_path=label_path,
            image_width=width,
            image_height=height,
        )

        return ImageRecord(
            image_path=image_path,
            label_path=label_path,
            width=width,
            height=height,
            annotations=annotations,
            is_dirty=False,
            predictions_loaded=False,
        )

    @staticmethod
    def _get_label_path(
        image_path: Path,
        image_directory: Path,
        label_directory: Path,
    ) -> Path:
        """
        Determine the YOLO label path corresponding to an image.

        Subdirectory structure is preserved when recursive image
        discovery is used.
        """
        relative_image_path = image_path.relative_to(
            image_directory
        )

        relative_label_path = relative_image_path.with_suffix(
            ".txt"
        )

        return label_directory / relative_label_path

    @staticmethod
    def _validate_classes(
        classes: list[ClassDefinition],
    ) -> None:
        """
        Ensure at least one valid class definition was supplied.
        """
        if not classes:
            raise ValueError(
                "At least one annotation class is required."
            )

        for class_definition in classes:
            if not isinstance(
                class_definition,
                ClassDefinition,
            ):
                raise TypeError(
                    "Classes must contain only "
                    "ClassDefinition objects."
                )