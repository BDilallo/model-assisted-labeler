from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from model_assisted_labeler.controllers.annotation_sources import (
    MODEL_SOURCE,
)
from model_assisted_labeler.controllers.session_context import SessionContext
from model_assisted_labeler.models.annotation_session import (
    AnnotationSession,
    ClassDefinition,
)
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.services.model_runner import (
    DetectionModelRunner,
)
from model_assisted_labeler.services.session_repository import (
    SessionRepository,
)


@dataclass(frozen=True)
class BatchAutoAnnotationResult:
    """Summarize one batch auto-annotation operation."""

    candidate_images: int
    processed_images: int
    saved_images: int
    rejected_images: int
    cancelled: bool


class ModelController:
    """Load detection models and run predictions against a session."""

    def __init__(
        self,
        context: SessionContext,
        model_runner: DetectionModelRunner,
        session_repository: SessionRepository,
    ) -> None:
        self._context = context
        self._model_runner = model_runner
        self._session_repository = session_repository

    @property
    def model_is_loaded(self) -> bool:
        return self._model_runner.is_loaded

    @property
    def model_path(self) -> Path | None:
        return self._model_runner.model_path

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

    def should_auto_predict_current_image(self) -> bool:
        image_record = self._context.current_image

        return bool(
            self.model_is_loaded
            and image_record is not None
            and image_record.annotations_loaded
            and not image_record.annotations
            and not image_record.in_annotation_pool
            and not image_record.predictions_loaded
            and not image_record.is_dirty
        )

    def batch_auto_annotation_candidate_count(self) -> int:
        """Return the number of clean, unsaved images eligible for batch."""
        if self._context.session is None:
            return 0

        return len(self._batch_auto_annotation_candidates())

    def batch_auto_annotate(
        self,
        confidence_threshold: float,
        progress_callback: (
            Callable[[int, int, str], None] | None
        ) = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> BatchAutoAnnotationResult:
        """Predict and save every eligible image above a confidence floor.

        Images already in the annotation pool and images containing
        unsaved work are never modified. An eligible image is saved only
        when at least one supported model box meets the supplied minimum
        confidence.
        """
        if not 0.0 <= confidence_threshold < 1.0:
            raise ValueError(
                "Confidence threshold must be at least 0 and less "
                "than 1."
            )

        session = self._context.require_session()
        definition = self._context.require_definition()

        if not self._model_runner.is_loaded:
            raise RuntimeError(
                "A detection model must be loaded before batch "
                "annotation."
            )

        self._validate_model_class_mapping(session)
        candidates = self._batch_auto_annotation_candidates()
        candidate_count = len(candidates)
        processed_count = 0
        saved_count = 0
        rejected_count = 0
        cancelled = False

        try:
            for image_record in candidates:
                if (
                    cancellation_requested is not None
                    and cancellation_requested()
                ):
                    cancelled = True
                    break

                try:
                    raw_predictions = self._model_runner.predict(
                        image_record.image_path,
                        confidence_threshold=confidence_threshold,
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Batch auto annotation failed for "
                        f"'{image_record.filename}': {error}"
                    ) from error

                predictions = [
                    box
                    for box in self._supported_predictions(
                        session,
                        raw_predictions,
                    )
                    if (
                        box.confidence is not None
                        and box.confidence >= confidence_threshold
                    )
                ]

                if predictions:
                    image_record.replace_annotations(predictions)
                    self._session_repository.save_image_to_pool(
                        definition,
                        image_record,
                        refresh_session_info=False,
                    )
                    image_record.mark_saved()
                    saved_count += 1
                else:
                    image_record.annotations = []
                    image_record.annotations_loaded = True
                    image_record.is_dirty = False
                    image_record.mark_predictions_loaded()
                    rejected_count += 1

                processed_count += 1

                if progress_callback is not None:
                    progress_callback(
                        processed_count,
                        candidate_count,
                        image_record.filename,
                    )

                if image_record is not session.current_image:
                    image_record.unload_annotations()

        finally:
            self._session_repository.save_session_info(definition)

        return BatchAutoAnnotationResult(
            candidate_images=candidate_count,
            processed_images=processed_count,
            saved_images=saved_count,
            rejected_images=rejected_count,
            cancelled=cancelled,
        )

    def predict_current_image(self) -> list[BoundingBox]:
        session = self._context.require_session()
        image_record = self._context.require_current_image()

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
            if box.source != MODEL_SOURCE
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
        session = self._context.require_session()
        image_record = self._context.require_current_image()

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

    def _batch_auto_annotation_candidates(
        self,
    ) -> list[ImageRecord]:
        """Return images safe for unattended prediction and saving."""
        session = self._context.require_session()

        return [
            image_record
            for image_record in session.images
            if (
                not image_record.in_annotation_pool
                and not image_record.is_dirty
                and not image_record.annotations
            )
        ]

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
