from model_assisted_labeler.models.annotation_session import AnnotationSession
from model_assisted_labeler.models.bounding_box import BoundingBox


class PredictionMixin:
    """Run the detection model against the current image."""

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
