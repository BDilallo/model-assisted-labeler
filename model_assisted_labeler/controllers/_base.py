from pathlib import Path

from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.services.dataset_exporter import DatasetExporter
from model_assisted_labeler.services.model_runner import (
    DetectionModelRunner,
)
from model_assisted_labeler.services.session_repository import (
    SessionRepository,
)


class _AnnotationControllerBase:
    """Shared state and low-level helpers used by every controller mixin."""

    MODEL_SOURCE = "model"
    MODEL_EDITED_SOURCE = "model_edited"
    EDITED_SOURCE = "edited"

    def __init__(
        self,
        session_builder: AnnotationSessionBuilder,
        annotation_store: YoloAnnotationStore,
        model_runner: DetectionModelRunner,
        session_repository: SessionRepository,
    ) -> None:
        self._session_builder = session_builder
        self._annotation_store = annotation_store
        self._model_runner = model_runner
        self._session_repository = session_repository
        self._dataset_exporter = DatasetExporter(
            session_repository=session_repository,
            annotation_store=annotation_store,
        )

        self._session: AnnotationSession | None = None
        self._session_definition: SessionDefinition | None = None

    @property
    def session(self) -> AnnotationSession | None:
        return self._session

    @property
    def session_definition(self) -> SessionDefinition | None:
        return self._session_definition

    @property
    def has_session(self) -> bool:
        return self._session is not None

    @property
    def current_image(self) -> ImageRecord | None:
        if self._session is None:
            return None

        return self._session.current_image

    @property
    def model_is_loaded(self) -> bool:
        return self._model_runner.is_loaded

    @property
    def model_path(self) -> Path | None:
        return self._model_runner.model_path

    @property
    def total_images_annotated(self) -> int:
        if self._session_definition is None:
            return 0

        return self._session_definition.total_images_annotated

    @property
    def default_dataset_export_directory(self) -> Path:
        """Return the default parent folder used by the export dialog."""
        return self._session_repository.workspace_root / "Exported Datasets"

    def _ensure_annotations_loaded(
        self,
        image_record: ImageRecord,
    ) -> None:
        if image_record.annotations_loaded:
            return

        definition = self._require_definition()
        in_pool = self._session_repository.image_is_in_pool(
            definition,
            image_record.image_path,
        )
        annotations = (
            self._session_repository.load_annotations(
                definition,
                image_record,
            )
            if in_pool
            else []
        )
        image_record.load_annotations(
            annotations=annotations,
            in_annotation_pool=in_pool,
        )

    def _require_session(self) -> AnnotationSession:
        if self._session is None:
            raise RuntimeError("No annotation session is currently open.")

        return self._session

    def _require_definition(self) -> SessionDefinition:
        if self._session_definition is None:
            raise RuntimeError("No saved session is currently open.")

        return self._session_definition

    def _require_current_image(self) -> ImageRecord:
        image_record = self._require_session().current_image

        if image_record is None:
            raise RuntimeError(
                "The annotation session contains no images."
            )

        self._ensure_annotations_loaded(image_record)
        return image_record

    @staticmethod
    def _edited_source_for(source: str) -> str:
        if source in {
            _AnnotationControllerBase.MODEL_SOURCE,
            _AnnotationControllerBase.MODEL_EDITED_SOURCE,
        }:
            return _AnnotationControllerBase.MODEL_EDITED_SOURCE

        return _AnnotationControllerBase.EDITED_SOURCE

    @staticmethod
    def _validate_class_id(
        session: AnnotationSession,
        class_id: int,
    ) -> None:
        if session.get_class(class_id) is None:
            raise ValueError(
                f"Class ID {class_id} is not defined in the session."
            )
