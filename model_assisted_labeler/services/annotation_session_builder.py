from collections.abc import Callable
from pathlib import Path

from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories.image_dimensions_cache_repository import (
    CachedImageDimensions,
    ImageDimensionsCacheRepository,
)
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
        dimensions_cache_repository: ImageDimensionsCacheRepository
        | None = None,
    ) -> None:
        self.image_service = image_service
        self.session_repository = session_repository
        self.dimensions_cache_repository = (
            dimensions_cache_repository
            or ImageDimensionsCacheRepository()
        )

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
        for large directories this can take a while. A per-session cache
        keyed by filename (validated against each file's current size
        and modification time) lets unchanged images skip that read on
        later loads; only images that are new or have changed since the
        cache was written are actually opened. ``progress_callback`` and
        ``cancellation_requested`` let a caller keep the UI responsive
        and let the user abort, mirroring dataset export/batch annotate.
        """
        image_paths = self.image_service.discover_images(
            directory=definition.image_directory,
            recursive=False,
        )

        cached_dimensions = self.dimensions_cache_repository.load(
            definition.session_directory
        )
        updated_cache: dict[str, CachedImageDimensions] = {}

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

            width, height = self._resolve_dimensions(
                image_path,
                cached_dimensions,
                updated_cache,
            )
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

        if updated_cache != cached_dimensions:
            try:
                self.dimensions_cache_repository.save(
                    definition.session_directory,
                    updated_cache,
                )
            except OSError:
                # The cache is a pure optimization; a failed write
                # should not prevent the session from opening.
                pass

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

    def _resolve_dimensions(
        self,
        image_path: Path,
        cached_dimensions: dict[str, CachedImageDimensions],
        updated_cache: dict[str, CachedImageDimensions],
    ) -> tuple[int, int]:
        """
        Return an image's dimensions, reusing the cache when the file's
        size and modification time have not changed since it was read.
        """
        stat_result = image_path.stat()
        cached_entry = cached_dimensions.get(image_path.name)

        if (
            cached_entry is not None
            and cached_entry.file_size == stat_result.st_size
            and cached_entry.modified_time == stat_result.st_mtime
        ):
            updated_cache[image_path.name] = cached_entry
            return cached_entry.width, cached_entry.height

        width, height = self.image_service.get_dimensions(image_path)
        updated_cache[image_path.name] = CachedImageDimensions(
            width=width,
            height=height,
            file_size=stat_result.st_size,
            modified_time=stat_result.st_mtime,
        )
        return width, height
