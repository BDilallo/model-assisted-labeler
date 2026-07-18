from dataclasses import replace
from pathlib import Path

from model_assisted_labeler.models.annotation_session import (
    AnnotationSession,
    ClassDefinition,
)
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.services.model_runner import (
    DetectionModelRunner,
)


class AnnotationController:
    """
    Coordinates the active annotation session.

    The controller handles session creation, model prediction,
    annotation changes, saving, and image navigation.

    It contains no GUI-specific code.
    """

    def __init__(
        self,
        session_builder: AnnotationSessionBuilder,
        annotation_store: YoloAnnotationStore,
        model_runner: DetectionModelRunner,
    ) -> None:
        """
        Store the services used by the annotation workflow.
        """
        self._session_builder = session_builder
        self._annotation_store = annotation_store
        self._model_runner = model_runner

        self._session: AnnotationSession | None = None

    @property
    def session(self) -> AnnotationSession | None:
        """Return the currently active annotation session."""
        return self._session

    @property
    def has_session(self) -> bool:
        """Return True when an annotation session is open."""
        return self._session is not None

    @property
    def current_image(self) -> ImageRecord | None:
        """
        Return the current image from the active session.

        Returns None when there is no session or the session contains
        no images.
        """
        if self._session is None:
            return None

        return self._session.current_image

    @property
    def model_is_loaded(self) -> bool:
        """Return True when a detection model is loaded."""
        return self._model_runner.is_loaded

    @property
    def model_path(self) -> Path | None:
        """Return the path of the currently loaded model."""
        return self._model_runner.model_path

    def load_model(
        self,
        model_path: Path,
    ) -> None:
        """
        Load a detection model through the configured model runner.
        """
        self._model_runner.load_model(model_path)

    def get_model_classes(self) -> list[ClassDefinition]:
        """
        Return the loaded model's classes as ClassDefinition objects.

        Raises:
            RuntimeError:
                If no model is loaded or the model provides no class
                definitions.
        """
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
            ClassDefinition(
                class_id=class_id,
                name=class_name,
            )
            for class_id, class_name in sorted(
                model_class_names.items()
            )
        ]

    def open_session(
        self,
        image_directory: Path,
        label_directory: Path,
        classes: list[ClassDefinition] | None = None,
        recursive: bool = False,
    ) -> AnnotationSession:
        """
        Build and activate an annotation session.

        If classes are not provided, they are taken from the loaded
        detection model.
        """
        if (
            self._session is not None
            and self._session.has_unsaved_changes()
        ):
            raise RuntimeError(
                "The current session contains unsaved changes."
            )

        resolved_classes = classes

        if resolved_classes is None:
            resolved_classes = self.get_model_classes()

        new_session = self._session_builder.build(
            image_directory=image_directory,
            label_directory=label_directory,
            classes=resolved_classes,
            recursive=recursive,
        )

        self._session = new_session

        return new_session

    def close_session(
        self,
        discard_unsaved_changes: bool = False,
    ) -> None:
        """
        Close the active session.

        By default, closing is blocked while unsaved changes exist.
        """
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

    def predict_current_image(
        self,
    ) -> list[BoundingBox]:
        """
        Run the loaded model on the current image.

        Existing untouched model predictions are removed before the
        new predictions are added. Imported, manually created, and
        edited annotations are preserved.
        """
        session = self._require_session()
        image_record = self._require_current_image()

        if not self._model_runner.is_loaded:
            raise RuntimeError(
                "A detection model must be loaded before prediction."
            )

        self._validate_model_class_mapping(session)

        predictions = self._model_runner.predict(
            image_record.image_path
        )

        self._validate_prediction_classes(
            session=session,
            predictions=predictions,
        )

        retained_annotations = [
            box
            for box in image_record.annotations
            if box.source != "model"
        ]

        image_record.replace_annotations(
            retained_annotations + predictions
        )

        image_record.mark_predictions_loaded()

        return predictions

    def replace_annotations_with_predictions(
        self,
    ) -> list[BoundingBox]:
        """
        Run prediction and replace every existing annotation.

        This is intentionally separate from predict_current_image()
        because replacing manual or imported annotations is destructive.
        """
        session = self._require_session()
        image_record = self._require_current_image()

        if not self._model_runner.is_loaded:
            raise RuntimeError(
                "A detection model must be loaded before prediction."
            )

        self._validate_model_class_mapping(session)

        predictions = self._model_runner.predict(
            image_record.image_path
        )

        self._validate_prediction_classes(
            session=session,
            predictions=predictions,
        )

        image_record.replace_annotations(predictions)
        image_record.mark_predictions_loaded()

        return predictions

    def add_annotation(
        self,
        box: BoundingBox,
    ) -> None:
        """
        Add a manually created annotation to the current image.
        """
        session = self._require_session()
        image_record = self._require_current_image()

        self._validate_class_id(
            session=session,
            class_id=box.class_id,
        )

        manual_box = replace(
            box,
            confidence=None,
            source="manual",
        )

        image_record.add_annotation(manual_box)

    def update_annotation(
        self,
        index: int,
        updated_box: BoundingBox,
    ) -> None:
        """
        Replace an annotation after the user moves or resizes it.

        Updated annotations are marked as edited so they are not
        removed when model prediction is run again.
        """
        session = self._require_session()
        image_record = self._require_current_image()

        self._validate_class_id(
            session=session,
            class_id=updated_box.class_id,
        )

        edited_box = replace(
            updated_box,
            source="edited",
        )

        image_record.update_annotation(
            index=index,
            updated_box=edited_box,
        )

    def change_annotation_class(
        self,
        index: int,
        class_id: int,
    ) -> None:
        """
        Change the class assigned to an existing annotation.
        """
        session = self._require_session()
        image_record = self._require_current_image()

        self._validate_class_id(
            session=session,
            class_id=class_id,
        )

        if index < 0 or index >= len(image_record.annotations):
            raise IndexError(
                f"Annotation index {index} is out of range."
            )

        existing_box = image_record.annotations[index]

        updated_box = replace(
            existing_box,
            class_id=class_id,
            source="edited",
        )

        image_record.update_annotation(
            index=index,
            updated_box=updated_box,
        )

    def remove_annotation(
        self,
        index: int,
    ) -> BoundingBox:
        """
        Remove and return an annotation from the current image.
        """
        image_record = self._require_current_image()

        return image_record.remove_annotation(index)

    def clear_current_annotations(self) -> None:
        """
        Remove every annotation from the current image.
        """
        image_record = self._require_current_image()

        image_record.clear_annotations()

    def save_current_image(self) -> None:
        """
        Save the current image's annotations to its YOLO label file.
        """
        image_record = self._require_current_image()

        self._annotation_store.save(
            label_path=image_record.label_path,
            annotations=image_record.annotations,
            image_width=image_record.width,
            image_height=image_record.height,
        )

        image_record.mark_saved()

    def save_all_changes(self) -> int:
        """
        Save every dirty image in the active session.

        Returns:
            The number of label files saved.
        """
        session = self._require_session()

        saved_count = 0

        for image_record in session.dirty_images():
            self._annotation_store.save(
                label_path=image_record.label_path,
                annotations=image_record.annotations,
                image_width=image_record.width,
                image_height=image_record.height,
            )

            image_record.mark_saved()
            saved_count += 1

        return saved_count

    def next_image(self) -> ImageRecord | None:
        """
        Move to and return the next image.

        Unsaved annotations remain stored in memory. They are not
        automatically written to disk.
        """
        session = self._require_session()

        return session.next_image()

    def previous_image(self) -> ImageRecord | None:
        """
        Move to and return the previous image.
        """
        session = self._require_session()

        return session.previous_image()

    def go_to_image(
        self,
        index: int,
    ) -> ImageRecord:
        """
        Select an image using its zero-based session index.
        """
        session = self._require_session()

        return session.go_to_image(index)

    def save_and_next(self) -> ImageRecord | None:
        """
        Save the current image and move to the next image.
        """
        session = self._require_session()

        self.save_current_image()

        return session.next_image()

    def has_unsaved_changes(self) -> bool:
        """
        Return True when the active session has unsaved changes.
        """
        if self._session is None:
            return False

        return self._session.has_unsaved_changes()

    def _require_session(self) -> AnnotationSession:
        """
        Return the active session or raise a clear error.
        """
        if self._session is None:
            raise RuntimeError(
                "No annotation session is currently open."
            )

        return self._session

    def _require_current_image(self) -> ImageRecord:
        """
        Return the current image or raise a clear error.
        """
        session = self._require_session()
        image_record = session.current_image

        if image_record is None:
            raise RuntimeError(
                "The annotation session contains no images."
            )

        return image_record

    @staticmethod
    def _validate_class_id(
        session: AnnotationSession,
        class_id: int,
    ) -> None:
        """
        Ensure a class ID exists in the current session.
        """
        if session.get_class(class_id) is None:
            raise ValueError(
                f"Class ID {class_id} is not defined "
                f"in the current session."
            )

    def _validate_prediction_classes(
        self,
        session: AnnotationSession,
        predictions: list[BoundingBox],
    ) -> None:
        """
        Ensure every predicted class exists in the session.
        """
        undefined_class_ids = sorted({
            box.class_id
            for box in predictions
            if session.get_class(box.class_id) is None
        })

        if undefined_class_ids:
            raise ValueError(
                "The model returned class IDs that are not "
                f"defined in the session: {undefined_class_ids}"
            )

    def _validate_model_class_mapping(
        self,
        session: AnnotationSession,
    ) -> None:
        """
        Ensure model class names agree with session class names.

        A matching numeric ID with a different name could silently
        create incorrectly labeled training data.
        """
        model_classes = self._model_runner.class_names

        if not model_classes:
            return

        mismatches: list[str] = []

        for class_id, model_name in model_classes.items():
            session_name = session.get_class_name(class_id)

            if session_name is None:
                mismatches.append(
                    f"{class_id}: model='{model_name}', "
                    "session=<missing>"
                )
                continue

            if session_name.casefold() != model_name.casefold():
                mismatches.append(
                    f"{class_id}: model='{model_name}', "
                    f"session='{session_name}'"
                )

        if mismatches:
            mismatch_text = "; ".join(mismatches)

            raise ValueError(
                "The model and session class mappings do not match: "
                f"{mismatch_text}"
            )