from model_assisted_labeler.controllers.session_context import SessionContext
from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.model_runner import (
    DetectionModelRunner,
)
from model_assisted_labeler.services.session_repository import (
    SessionRepository,
)


class SessionController:
    """Open, close, and navigate an annotation session."""

    def __init__(
        self,
        context: SessionContext,
        session_builder: AnnotationSessionBuilder,
        model_runner: DetectionModelRunner,
        session_repository: SessionRepository,
    ) -> None:
        self._context = context
        self._session_builder = session_builder
        self._model_runner = model_runner
        self._session_repository = session_repository

    def open_session_definition(
        self,
        definition: SessionDefinition,
    ) -> AnnotationSession:
        if self._context.has_unsaved_changes():
            raise RuntimeError(
                "The current session contains unsaved changes."
            )

        model_path = definition.primary_model_path

        if model_path is not None:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Model file does not exist: {model_path}"
                )

            if self._model_runner.model_path != model_path:
                self._model_runner.load_model(model_path)
        else:
            self._model_runner.unload_model()

        new_session = self._session_builder.build(definition)

        self._context.set_session(new_session, definition)
        self.prepare_current_image()
        return new_session

    def close_session(
        self,
        discard_unsaved_changes: bool = False,
    ) -> None:
        session = self._context.session

        if session is None:
            return

        if (
            session.has_unsaved_changes()
            and not discard_unsaved_changes
        ):
            raise RuntimeError(
                "The session contains unsaved changes."
            )

        self._context.clear_session()

    def prepare_current_image(self) -> ImageRecord | None:
        """Load the current image's saved boxes and persist its position."""
        session = self._context.session

        if session is None:
            return None

        image_record = session.current_image
        definition = self._context.require_definition()

        if image_record is None:
            self._session_repository.update_last_image(definition, None)
            return None

        self._context.ensure_annotations_loaded(image_record)
        self._session_repository.update_last_image(
            definition,
            image_record.filename,
        )
        return image_record

    def prefetch_nearby_annotations(self, radius: int = 5) -> None:
        """Cache clean annotations around the current image."""
        if radius < 0:
            raise ValueError("Prefetch radius cannot be negative.")

        session = self._context.require_session()

        if not session.images:
            return

        start_index = max(0, session.current_index - radius)
        end_index = min(
            len(session.images) - 1,
            session.current_index + radius,
        )
        retained_indexes = set(range(start_index, end_index + 1))

        for index in retained_indexes:
            self._context.ensure_annotations_loaded(session.images[index])

        for index, image_record in enumerate(session.images):
            if index not in retained_indexes:
                image_record.unload_annotations()

    def prefetch_annotation_indexes(
        self,
        image_indexes: list[int],
    ) -> None:
        """Cache annotations for selected source indexes only.

        Dirty records are still protected by ``ImageRecord`` and are not
        discarded when they fall outside the requested review window.
        """
        session = self._context.require_session()
        retained_indexes = set(image_indexes)

        if session.images:
            retained_indexes.add(session.current_index)

        for index in retained_indexes:
            if index < 0 or index >= len(session.images):
                raise IndexError(f"Image index {index} is out of range.")

            self._context.ensure_annotations_loaded(session.images[index])

        for index, image_record in enumerate(session.images):
            if index not in retained_indexes:
                image_record.unload_annotations()

    def next_image(self) -> ImageRecord | None:
        session = self._context.require_session()
        session.next_image()
        return self.prepare_current_image()

    def previous_image(self) -> ImageRecord | None:
        session = self._context.require_session()
        session.previous_image()
        return self.prepare_current_image()

    def go_to_image(self, index: int) -> ImageRecord:
        session = self._context.require_session()
        session.go_to_image(index)
        image_record = self.prepare_current_image()

        if image_record is None:
            raise RuntimeError("The annotation session contains no images.")

        return image_record
