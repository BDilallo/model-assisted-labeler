from dataclasses import replace
from pathlib import Path

from model_assisted_labeler.models.annotation_session import (
    AnnotationSession,
    ClassDefinition,
)
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.services.model_runner import (
    DetectionModelRunner,
)
from model_assisted_labeler.services.session_repository import (
    SessionRepository,
)


class AnnotationController:
    """Coordinate session state, persistence, and model prediction."""

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

    def load_model(self, model_path: Path) -> None:
        self._model_runner.load_model(model_path)

    def clear_model(self) -> None:
        self._model_runner.unload_model()

    def inspect_model_classes(
        self,
        model_path: Path,
    ) -> list[ClassDefinition]:
        """Load a candidate model and return its class definitions."""
        self.load_model(model_path)
        return self.get_model_classes()

    def get_model_classes(self) -> list[ClassDefinition]:
        if not self._model_runner.is_loaded:
            raise RuntimeError(
                "A model must be loaded before reading its classes."
            )

        model_class_names = self._model_runner.class_names

        if not model_class_names:
            raise RuntimeError(
                "The loaded model does not provide class names."
            )

        return [
            ClassDefinition(class_id=class_id, name=class_name)
            for class_id, class_name in sorted(model_class_names.items())
        ]

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

    def should_auto_predict_current_image(self) -> bool:
        image_record = self.current_image

        return bool(
            self.model_is_loaded
            and image_record is not None
            and image_record.annotations_loaded
            and not image_record.annotations
            and not image_record.in_annotation_pool
            and not image_record.predictions_loaded
            and not image_record.is_dirty
        )

    def predict_current_image(self) -> list[BoundingBox]:
        session = self._require_session()
        image_record = self._require_current_image()

        if not self._model_runner.is_loaded:
            raise RuntimeError(
                "A detection model must be loaded before prediction."
            )

        self._validate_model_class_mapping(session)
        predictions = self._supported_predictions(
            session,
            self._model_runner.predict(image_record.image_path),
        )

        retained_annotations = [
            box
            for box in image_record.annotations
            if box.source != self.MODEL_SOURCE
        ]
        combined_annotations = retained_annotations + predictions

        if combined_annotations:
            image_record.replace_annotations(combined_annotations)
        else:
            image_record.annotations = []
            image_record.annotations_loaded = True
            image_record.is_dirty = False

        image_record.mark_predictions_loaded()
        return predictions

    def replace_annotations_with_predictions(self) -> list[BoundingBox]:
        session = self._require_session()
        image_record = self._require_current_image()

        if not self._model_runner.is_loaded:
            raise RuntimeError(
                "A detection model must be loaded before prediction."
            )

        self._validate_model_class_mapping(session)
        predictions = self._supported_predictions(
            session,
            self._model_runner.predict(image_record.image_path),
        )

        if predictions:
            image_record.replace_annotations(predictions)
        else:
            image_record.annotations = []
            image_record.annotations_loaded = True
            image_record.is_dirty = image_record.in_annotation_pool

        image_record.mark_predictions_loaded()
        return predictions

    def add_annotation(self, box: BoundingBox) -> None:
        session = self._require_session()
        image_record = self._require_current_image()
        self._validate_class_id(session, box.class_id)

        image_record.add_annotation(
            replace(box, confidence=None, source="manual")
        )

    def update_annotation(
        self,
        index: int,
        updated_box: BoundingBox,
    ) -> None:
        session = self._require_session()
        image_record = self._require_current_image()
        self._validate_class_id(session, updated_box.class_id)

        if index < 0 or index >= len(image_record.annotations):
            raise IndexError(f"Annotation index {index} is out of range.")

        existing_box = image_record.annotations[index]
        image_record.update_annotation(
            index=index,
            updated_box=replace(
                updated_box,
                confidence=existing_box.confidence,
                source=self._edited_source_for(existing_box.source),
            ),
        )

    def change_annotation_class(
        self,
        index: int,
        class_id: int,
    ) -> None:
        session = self._require_session()
        image_record = self._require_current_image()
        self._validate_class_id(session, class_id)

        if index < 0 or index >= len(image_record.annotations):
            raise IndexError(f"Annotation index {index} is out of range.")

        existing_box = image_record.annotations[index]
        image_record.update_annotation(
            index=index,
            updated_box=replace(
                existing_box,
                class_id=class_id,
                source=self._edited_source_for(existing_box.source),
            ),
        )

    def remove_annotation(self, index: int) -> BoundingBox:
        return self._require_current_image().remove_annotation(index)

    def clear_current_annotations(self) -> None:
        self._require_current_image().clear_annotations()

    def save_current_image(self) -> None:
        image_record = self._require_current_image()
        definition = self._require_definition()

        if not image_record.annotations:
            raise ValueError(
                "An image must contain at least one box before it can "
                "be saved."
            )

        self._session_repository.save_image_to_pool(
            definition,
            image_record,
        )
        image_record.mark_saved()

    def save_all_changes(self) -> int:
        session = self._require_session()
        definition = self._require_definition()
        dirty_images = session.dirty_images()

        empty_pooled_images = [
            image
            for image in dirty_images
            if not image.annotations and image.in_annotation_pool
        ]

        if empty_pooled_images:
            raise RuntimeError(
                "One or more saved images now contain no boxes. Use "
                "Remove from Annotation Pool for those images, or "
                "discard the changes."
            )

        saved_count = 0

        for image_record in dirty_images:
            if not image_record.annotations:
                image_record.is_dirty = False
                continue

            self._session_repository.save_image_to_pool(
                definition,
                image_record,
            )
            image_record.mark_saved()
            saved_count += 1

        return saved_count

    def remove_current_from_annotation_pool(self) -> None:
        image_record = self._require_current_image()
        definition = self._require_definition()

        if not image_record.in_annotation_pool:
            raise RuntimeError(
                "The current image is not in the annotation pool."
            )

        self._session_repository.remove_image_from_pool(
            definition,
            image_record.image_path,
        )
        image_record.mark_removed_from_pool()

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

    def add_class(self, class_name: str) -> ClassDefinition:
        session = self._require_session()
        definition = self._require_definition()
        cleaned_name = class_name.strip()

        if not cleaned_name:
            raise ValueError("Class name cannot be empty.")

        if session.get_class_by_name(cleaned_name) is not None:
            raise ValueError(f"Class '{cleaned_name}' already exists.")

        model_class = self._model_class_by_name(cleaned_name)

        if model_class is not None:
            class_id = model_class.class_id
            existing_id_class = session.get_class(class_id)

            if existing_id_class is not None:
                raise ValueError(
                    "The model class ID is already used by "
                    f"'{existing_id_class.name}'."
                )
        else:
            reserved_ids = set(self._model_runner.class_names.keys())
            used_ids = {item.class_id for item in session.classes}
            class_id = max(definition.next_class_id, 0)

            while class_id in reserved_ids or class_id in used_ids:
                class_id += 1

            definition.next_class_id = class_id + 1

        class_definition = ClassDefinition(
            class_id=class_id,
            name=(model_class.name if model_class else cleaned_name),
        )
        session.add_class(class_definition)
        definition.classes = list(session.classes)
        definition.next_class_id = max(
            definition.next_class_id,
            class_id + 1,
        )
        self._session_repository.save_classes(definition)
        return class_definition

    def class_usage_filenames(self, class_id: int) -> list[str]:
        session = self._require_session()
        self._validate_class_id(session, class_id)
        return self._session_repository.class_usage_filenames(
            self._require_definition(),
            class_id,
        )

    def delete_class(
        self,
        class_id: int,
        mode: str,
    ) -> ClassDefinition:
        """
        Delete a class and update persisted annotations.

        ``mode='remove'`` removes only boxes using the class.
        ``mode='delete'`` removes every pooled image containing it.
        """
        if mode not in {"remove", "delete"}:
            raise ValueError("Class deletion mode must be remove or delete.")

        session = self._require_session()
        definition = self._require_definition()
        class_definition = session.get_class(class_id)

        if class_definition is None:
            raise ValueError(f"Class ID {class_id} is not defined.")

        if len(session.classes) == 1:
            raise ValueError(
                "A session must contain at least one class. Add another "
                "class before deleting this one."
            )

        affected_stems = (
            self._session_repository.remove_class_from_pool_annotations(
                definition=definition,
                class_id=class_id,
                delete_referenced_images=(mode == "delete"),
            )
        )
        normalized_affected = {
            stem.casefold() for stem in affected_stems
        }

        for image_record in session.images:
            stem_is_affected = (
                image_record.image_path.stem.casefold()
                in normalized_affected
            )

            if mode == "delete" and stem_is_affected:
                image_record.mark_removed_from_pool()
                continue

            if not image_record.annotations_loaded:
                image_record.in_annotation_pool = (
                    self._session_repository.image_is_in_pool(
                        definition,
                        image_record.image_path,
                    )
                )
                continue

            filtered_annotations = [
                box
                for box in image_record.annotations
                if box.class_id != class_id
            ]
            changed = (
                len(filtered_annotations)
                != len(image_record.annotations)
            )

            if not changed:
                image_record.in_annotation_pool = (
                    self._session_repository.image_is_in_pool(
                        definition,
                        image_record.image_path,
                    )
                )
                continue

            was_dirty = image_record.is_dirty
            was_in_pool = image_record.in_annotation_pool
            image_record.annotations = filtered_annotations
            image_record.annotations_loaded = True
            image_record.predictions_loaded = False

            if was_dirty and was_in_pool:
                # A confirmed class deletion is authoritative for the
                # dataset. Preserve any other unsaved box edits while
                # keeping the pooled files consistent with memory.
                if filtered_annotations:
                    self._session_repository.save_image_to_pool(
                        definition,
                        image_record,
                    )
                    image_record.mark_saved()
                else:
                    self._session_repository.remove_image_from_pool(
                        definition,
                        image_record.image_path,
                    )
                    image_record.mark_removed_from_pool()

            elif was_dirty:
                image_record.in_annotation_pool = False
                image_record.is_dirty = bool(filtered_annotations)

            else:
                image_record.in_annotation_pool = (
                    self._session_repository.image_is_in_pool(
                        definition,
                        image_record.image_path,
                    )
                )
                image_record.is_dirty = False

        removed_class = session.remove_class(class_id)
        definition.classes = list(session.classes)
        self._session_repository.save_classes(definition)
        return removed_class

    def has_unsaved_changes(self) -> bool:
        return bool(
            self._session is not None
            and self._session.has_unsaved_changes()
        )

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

    def _supported_predictions(
        self,
        session: AnnotationSession,
        predictions: list[BoundingBox],
    ) -> list[BoundingBox]:
        """Ignore model classes intentionally removed from the session."""
        return [
            box
            for box in predictions
            if session.get_class(box.class_id) is not None
        ]

    def _validate_model_class_mapping(
        self,
        session: AnnotationSession,
    ) -> None:
        model_classes = self._model_runner.class_names
        mismatches: list[str] = []

        for class_id, model_name in model_classes.items():
            session_class_at_id = session.get_class(class_id)
            session_class_by_name = session.get_class_by_name(model_name)

            if (
                session_class_at_id is not None
                and session_class_at_id.name.casefold()
                != model_name.casefold()
            ):
                mismatches.append(
                    f"ID {class_id}: model='{model_name}', "
                    f"session='{session_class_at_id.name}'"
                )

            if (
                session_class_by_name is not None
                and session_class_by_name.class_id != class_id
            ):
                mismatches.append(
                    f"Name '{model_name}': model ID={class_id}, "
                    f"session ID={session_class_by_name.class_id}"
                )

        if mismatches:
            raise ValueError(
                "The model and session class mappings conflict: "
                + "; ".join(mismatches)
            )

    def _model_class_by_name(
        self,
        class_name: str,
    ) -> ClassDefinition | None:
        normalized_name = class_name.strip().casefold()

        for class_id, model_name in self._model_runner.class_names.items():
            if model_name.casefold() == normalized_name:
                return ClassDefinition(class_id, model_name)

        return None

    @staticmethod
    def _edited_source_for(source: str) -> str:
        if source in {
            AnnotationController.MODEL_SOURCE,
            AnnotationController.MODEL_EDITED_SOURCE,
        }:
            return AnnotationController.MODEL_EDITED_SOURCE

        return AnnotationController.EDITED_SOURCE

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
    def _validate_class_id(
        session: AnnotationSession,
        class_id: int,
    ) -> None:
        if session.get_class(class_id) is None:
            raise ValueError(
                f"Class ID {class_id} is not defined in the session."
            )
