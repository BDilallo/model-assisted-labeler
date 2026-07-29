from collections.abc import Callable

from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories.session_repository import (
    SessionRepository,
)
from model_assisted_labeler.services.image_service import ImageService

ProgressCallback = Callable[[int, int, str], None]
CancellationCallback = Callable[[], bool]


class SessionBuildCancelled(Exception):
    """Raised when the caller reports cancellation mid-build."""


class AnnotationSessionBuilder:
    """Build a lazy in-memory session from a saved definition."""

    def __init__(
        self,
        image_service: ImageService,
        session_repository: SessionRepository,
    ) -> None:
        self.image_service = image_service
        self.session_repository = session_repository

    def build(
        self,
        definition: SessionDefinition,
        progress_callback: ProgressCallback | None = None,
        cancellation_requested: CancellationCallback | None = None,
    ) -> AnnotationSession:
        """
        Discover only top-level source images.

        Annotation files are intentionally not loaded here. The
        controller loads the current image immediately and prefetches a
        small surrounding window after navigation settles.

        Reading each image's dimensions requires opening every file, so
        for large directories this can take a while. ``progress_callback``
        and ``cancellation_requested`` let a caller keep the UI responsive
        and let the user abort, mirroring dataset export/batch annotate.
        """
        image_paths = self.image_service.discover_images(
            directory=definition.image_directory,
            recursive=False,
        )

        image_records: list[ImageRecord] = []
        total_images = len(image_paths)

        for index, image_path in enumerate(image_paths):
            if (
                cancellation_requested is not None
                and cancellation_requested()
            ):
                raise SessionBuildCancelled()

            if progress_callback is not None:
                progress_callback(index, total_images, image_path.name)

            width, height = self.image_service.get_dimensions(image_path)
            label_path = self.session_repository.annotation_path_for(
                definition,
                image_path,
            )

            image_records.append(
                ImageRecord(
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    annotations=[],
                    is_dirty=False,
                    predictions_loaded=False,
                    annotations_loaded=False,
                    in_annotation_pool=(
                        self.session_repository.image_is_in_pool(
                            definition,
                            image_path,
                        )
                    ),
                )
            )

        current_index = 0
        target_filename = definition.last_image_loaded

        if target_filename:
            normalized_target = target_filename.casefold()

            for index, image_record in enumerate(image_records):
                if image_record.filename.casefold() == normalized_target:
                    current_index = index
                    break

        return AnnotationSession(
            name=definition.name,
            session_directory=definition.session_directory,
            image_directory=definition.image_directory,
            label_directory=(
                self.session_repository.annotations_directory(
                    definition.session_directory
                )
            ),
            classes=list(definition.classes),
            images=image_records,
            current_index=current_index,
        )
