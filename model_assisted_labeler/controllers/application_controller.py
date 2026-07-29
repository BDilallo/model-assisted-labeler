from collections.abc import Callable
from pathlib import Path

from model_assisted_labeler.controllers.annotation_edit_controller import (
    AnnotationEditController,
)
from model_assisted_labeler.controllers.annotation_sources import (
    EDITED_SOURCE,
    MODEL_EDITED_SOURCE,
    MODEL_SOURCE,
)
from model_assisted_labeler.controllers.class_controller import (
    ClassController,
)
from model_assisted_labeler.controllers.dataset_export_controller import (
    DatasetExportController,
)
from model_assisted_labeler.controllers.image_filter_controller import (
    ImageFilterController,
)
from model_assisted_labeler.controllers.model_controller import (
    BatchAutoAnnotationResult,
    ModelController,
)
from model_assisted_labeler.controllers.session_context import (
    SessionContext,
)
from model_assisted_labeler.controllers.session_controller import (
    SessionController,
)
from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.class_definition import ClassDefinition
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.repositories.session_repository import (
    SessionRepository,
)
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.dataset_export_service import (
    CancellationCallback,
    DatasetExportResult,
    DatasetExportService,
    DatasetExportSettings,
    ProgressCallback,
)
from model_assisted_labeler.services.model_runner import (
    DetectionModelRunner,
)

__all__ = ["AnnotationController", "BatchAutoAnnotationResult"]


class AnnotationController:
    """Coordinate session state, persistence, and model prediction.

    A thin facade that delegates to the session, model, editing, class,
    and image-filter controllers so callers keep a single entry point.
    """

    MODEL_SOURCE = MODEL_SOURCE
    MODEL_EDITED_SOURCE = MODEL_EDITED_SOURCE
    EDITED_SOURCE = EDITED_SOURCE

    def __init__(
        self,
        session_builder: AnnotationSessionBuilder,
        annotation_store: YoloAnnotationStore,
        model_runner: DetectionModelRunner,
        session_repository: SessionRepository,
    ) -> None:
        self._annotation_store = annotation_store

        context = SessionContext(session_repository)
        self._context = context

        self._session_controller = SessionController(
            context=context,
            session_builder=session_builder,
            model_runner=model_runner,
            session_repository=session_repository,
        )
        self._model_controller = ModelController(
            context=context,
            model_runner=model_runner,
            session_repository=session_repository,
        )
        self._edit_controller = AnnotationEditController(
            context=context,
            session_repository=session_repository,
        )
        self._class_controller = ClassController(
            context=context,
            model_runner=model_runner,
            session_repository=session_repository,
        )
        self._filter_controller = ImageFilterController(context=context)
        self._export_controller = DatasetExportController(
            context=context,
            export_service=DatasetExportService(),
        )

    # -- session state --------------------------------------------------

    @property
    def session(self) -> AnnotationSession | None:
        return self._context.session

    @property
    def session_definition(self) -> SessionDefinition | None:
        return self._context.session_definition

    @property
    def has_session(self) -> bool:
        return self._context.has_session

    @property
    def current_image(self) -> ImageRecord | None:
        return self._context.current_image

    @property
    def total_images_annotated(self) -> int:
        return self._context.total_images_annotated

    def has_unsaved_changes(self) -> bool:
        return self._context.has_unsaved_changes()

    # -- model ------------------------------------------------------------

    @property
    def model_is_loaded(self) -> bool:
        return self._model_controller.model_is_loaded

    @property
    def model_path(self) -> Path | None:
        return self._model_controller.model_path

    def load_model(self, model_path: Path) -> None:
        self._model_controller.load_model(model_path)

    def clear_model(self) -> None:
        self._model_controller.clear_model()

    def inspect_model_classes(
        self,
        model_path: Path,
    ) -> list[ClassDefinition]:
        return self._model_controller.inspect_model_classes(model_path)

    def get_model_classes(self) -> list[ClassDefinition]:
        return self._model_controller.get_model_classes()

    def should_auto_predict_current_image(self) -> bool:
        return self._model_controller.should_auto_predict_current_image()

    def batch_auto_annotation_candidate_count(self) -> int:
        return (
            self._model_controller.batch_auto_annotation_candidate_count()
        )

    @property
    def cuda_is_available(self) -> bool:
        return self._model_controller.cuda_is_available

    def batch_auto_annotate(
        self,
        confidence_threshold: float,
        batch_size: int = ModelController.DEFAULT_BATCH_SIZE,
        device: str | int | None = None,
        progress_callback: (
            Callable[[int, int, str], None] | None
        ) = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> BatchAutoAnnotationResult:
        return self._model_controller.batch_auto_annotate(
            confidence_threshold,
            batch_size=batch_size,
            device=device,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )

    def predict_current_image(self) -> list[BoundingBox]:
        return self._model_controller.predict_current_image()

    def replace_annotations_with_predictions(self) -> list[BoundingBox]:
        return self._model_controller.replace_annotations_with_predictions()

    # -- session lifecycle & navigation ------------------------------------

    def open_session_definition(
        self,
        definition: SessionDefinition,
        progress_callback: (
            Callable[[int, int, str], None] | None
        ) = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> AnnotationSession:
        return self._session_controller.open_session_definition(
            definition,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )

    def close_session(
        self,
        discard_unsaved_changes: bool = False,
    ) -> None:
        self._session_controller.close_session(discard_unsaved_changes)

    def prepare_current_image(self) -> ImageRecord | None:
        return self._session_controller.prepare_current_image()

    def prefetch_nearby_annotations(self, radius: int = 5) -> None:
        self._session_controller.prefetch_nearby_annotations(radius)

    def prefetch_annotation_indexes(
        self,
        image_indexes: list[int],
    ) -> None:
        self._session_controller.prefetch_annotation_indexes(image_indexes)

    def next_image(self) -> ImageRecord | None:
        return self._session_controller.next_image()

    def previous_image(self) -> ImageRecord | None:
        return self._session_controller.previous_image()

    def go_to_image(self, index: int) -> ImageRecord:
        return self._session_controller.go_to_image(index)

    def save_and_next(self) -> ImageRecord | None:
        self._edit_controller.save_current_image()
        return self._session_controller.next_image()

    # -- annotation editing -------------------------------------------------

    def add_annotation(self, box: BoundingBox) -> None:
        self._edit_controller.add_annotation(box)

    def update_annotation(
        self,
        index: int,
        updated_box: BoundingBox,
    ) -> None:
        self._edit_controller.update_annotation(index, updated_box)

    def change_annotation_class(
        self,
        index: int,
        class_id: int,
    ) -> None:
        self._edit_controller.change_annotation_class(index, class_id)

    def remove_annotation(self, index: int) -> BoundingBox:
        return self._edit_controller.remove_annotation(index)

    def clear_current_annotations(self) -> None:
        self._edit_controller.clear_current_annotations()

    def save_current_image(self) -> None:
        self._edit_controller.save_current_image()

    def save_all_changes(self) -> int:
        return self._edit_controller.save_all_changes()

    def remove_current_from_annotation_pool(self) -> None:
        self._edit_controller.remove_current_from_annotation_pool()

    def clear_all_annotations(self) -> None:
        self._edit_controller.clear_all_annotations()

    # -- classes --------------------------------------------------------------

    def add_class(self, class_name: str) -> ClassDefinition:
        return self._class_controller.add_class(class_name)

    def class_usage_filenames(self, class_id: int) -> list[str]:
        return self._class_controller.class_usage_filenames(class_id)

    def delete_class(
        self,
        class_id: int,
        mode: str,
    ) -> ClassDefinition:
        return self._class_controller.delete_class(class_id, mode)

    # -- image filters ------------------------------------------------------

    def image_indexes_matching(
        self,
        filter_key: str,
        confidence_threshold: float = 0.8,
        class_id: int | None = None,
    ) -> list[int]:
        return self._filter_controller.image_indexes_matching(
            filter_key,
            confidence_threshold=confidence_threshold,
            class_id=class_id,
        )

    def image_index_matches_filter(
        self,
        image_index: int,
        filter_key: str,
        confidence_threshold: float = 0.8,
        class_id: int | None = None,
    ) -> bool:
        return self._filter_controller.image_index_matches_filter(
            image_index,
            filter_key,
            confidence_threshold=confidence_threshold,
            class_id=class_id,
        )

    # -- dataset export ------------------------------------------------------

    @property
    def pooled_image_count(self) -> int:
        return self._export_controller.pooled_image_count

    def compute_export_split(
        self,
        settings: DatasetExportSettings,
    ) -> tuple[int, int]:
        return self._export_controller.compute_export_split(settings)

    def export_dataset(
        self,
        settings: DatasetExportSettings,
        progress_callback: ProgressCallback | None = None,
        cancellation_requested: CancellationCallback | None = None,
    ) -> DatasetExportResult:
        return self._export_controller.export_dataset(
            settings,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )
