from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories.session_repository import (
    SessionRepository,
)


def validate_class_id(session: AnnotationSession, class_id: int) -> None:
    if session.get_class(class_id) is None:
        raise ValueError(
            f"Class ID {class_id} is not defined in the session."
        )


class SessionContext:
    """Hold the active session/session-definition state shared by the
    controllers that operate on an open annotation session.
    """

    def __init__(self, session_repository: SessionRepository) -> None:
        self._session_repository = session_repository
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
    def total_images_annotated(self) -> int:
        if self._session_definition is None:
            return 0

        return self._session_definition.total_images_annotated

    def set_session(
        self,
        session: AnnotationSession,
        definition: SessionDefinition,
    ) -> None:
        self._session = session
        self._session_definition = definition

    def clear_session(self) -> None:
        self._session = None
        self._session_definition = None

    def has_unsaved_changes(self) -> bool:
        return bool(
            self._session is not None
            and self._session.has_unsaved_changes()
        )

    def require_session(self) -> AnnotationSession:
        if self._session is None:
            raise RuntimeError("No annotation session is currently open.")

        return self._session

    def require_definition(self) -> SessionDefinition:
        if self._session_definition is None:
            raise RuntimeError("No saved session is currently open.")

        return self._session_definition

    def require_current_image(self) -> ImageRecord:
        image_record = self.require_session().current_image

        if image_record is None:
            raise RuntimeError(
                "The annotation session contains no images."
            )

        self.ensure_annotations_loaded(image_record)
        return image_record

    def ensure_annotations_loaded(self, image_record: ImageRecord) -> None:
        if image_record.annotations_loaded:
            return

        definition = self.require_definition()
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
