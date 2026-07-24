from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition


class SessionLifecycleMixin:
    """Open, close, and navigate annotation sessions."""

    def open_session_definition(
        self,
        definition: SessionDefinition,
    ) -> AnnotationSession:
        if self.has_unsaved_changes():
            raise RuntimeError(
                "The current session contains unsaved changes."
            )

        model_path = definition.primary_model_path

        if model_path is not None:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Model file does not exist: {model_path}"
                )

            if self.model_path != model_path:
                self.load_model(model_path)
        else:
            self.clear_model()

        new_session = self._session_builder.build(definition)

        self._session_definition = definition
        self._session = new_session
        self.prepare_current_image()
        return new_session

    def close_session(
        self,
        discard_unsaved_changes: bool = False,
    ) -> None:
        if self._session is None:
            return

        if (
            self._session.has_unsaved_changes()
            and not discard_unsaved_changes
        ):
            raise RuntimeError(
                "The session contains unsaved changes."
            )

        self._session = None
        self._session_definition = None

    def prepare_current_image(self) -> ImageRecord | None:
        """Load the current image's saved boxes and persist its position."""
        if self._session is None:
            return None

        image_record = self._session.current_image
        definition = self._require_definition()

        if image_record is None:
            self._session_repository.update_last_image(definition, None)
            return None

        self._ensure_annotations_loaded(image_record)
        self._session_repository.update_last_image(
            definition,
            image_record.filename,
        )
        return image_record

    def prefetch_nearby_annotations(self, radius: int = 5) -> None:
        """Cache clean annotations around the current image."""
        if radius < 0:
            raise ValueError("Prefetch radius cannot be negative.")

        session = self._require_session()

        if not session.images:
            return

        start_index = max(0, session.current_index - radius)
        end_index = min(
            len(session.images) - 1,
            session.current_index + radius,
        )
        retained_indexes = set(range(start_index, end_index + 1))

        for index in retained_indexes:
            self._ensure_annotations_loaded(session.images[index])

        for index, image_record in enumerate(session.images):
            if index not in retained_indexes:
                image_record.unload_annotations()

    def next_image(self) -> ImageRecord | None:
        session = self._require_session()
        session.next_image()
        return self.prepare_current_image()

    def previous_image(self) -> ImageRecord | None:
        session = self._require_session()
        session.previous_image()
        return self.prepare_current_image()

    def go_to_image(self, index: int) -> ImageRecord:
        session = self._require_session()
        session.go_to_image(index)
        image_record = self.prepare_current_image()

        if image_record is None:
            raise RuntimeError("The annotation session contains no images.")

        return image_record

    def save_and_next(self) -> ImageRecord | None:
        session = self._require_session()
        self.save_current_image()
        session.next_image()
        return self.prepare_current_image()

    def has_unsaved_changes(self) -> bool:
        return bool(
            self._session is not None
            and self._session.has_unsaved_changes()
        )
