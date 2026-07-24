from model_assisted_labeler.controllers.annotation_sources import (
    EDITED_SOURCE,
    MODEL_EDITED_SOURCE,
    MODEL_SOURCE,
)
from model_assisted_labeler.controllers.session_context import (
    SessionContext,
    validate_class_id,
)
from model_assisted_labeler.models.image_record import ImageRecord


class ImageFilterController:
    """Answer which session images match the review filter controls."""

    IMAGE_FILTER_KEYS = {
        "all",
        "unsaved",
        "saved",
        "no_boxes",
        "unsaved_changes",
        "confidence_below",
        "confidence_at_or_above",
        "manual",
        "model_only",
        "edited_model",
        "single_box",
        "multiple_boxes",
        "missing_class",
    }

    FILTERS_REQUIRING_ANNOTATIONS = {
        "no_boxes",
        "confidence_below",
        "confidence_at_or_above",
        "manual",
        "model_only",
        "edited_model",
        "single_box",
        "multiple_boxes",
        "missing_class",
    }

    def __init__(self, context: SessionContext) -> None:
        self._context = context

    def image_indexes_matching(
        self,
        filter_key: str,
        confidence_threshold: float = 0.8,
        class_id: int | None = None,
    ) -> list[int]:
        """Return source-image indexes matching the review controls.

        ``filter_key`` selects the primary filter. A selected ``class_id``
        additionally narrows most filters to images containing that class.
        Confidence filters use the lowest stored model confidence, limited
        to the selected class when one is supplied.
        """
        normalized_filter = self._validate_image_filter_arguments(
            filter_key=filter_key,
            confidence_threshold=confidence_threshold,
            class_id=class_id,
        )
        session = self._context.require_session()
        matching_indexes: list[int] = []
        needs_annotations = (
            normalized_filter in self.FILTERS_REQUIRING_ANNOTATIONS
            or class_id is not None
        )

        for index, image_record in enumerate(session.images):
            loaded_for_filter = False

            if needs_annotations and not image_record.annotations_loaded:
                self._context.ensure_annotations_loaded(image_record)
                loaded_for_filter = True

            if self._image_matches_filter(
                image_record=image_record,
                filter_key=normalized_filter,
                confidence_threshold=confidence_threshold,
                class_id=class_id,
            ):
                matching_indexes.append(index)

            if (
                loaded_for_filter
                and image_record is not session.current_image
            ):
                image_record.unload_annotations()

        return matching_indexes

    def image_index_matches_filter(
        self,
        image_index: int,
        filter_key: str,
        confidence_threshold: float = 0.8,
        class_id: int | None = None,
    ) -> bool:
        """Return whether one source-image index matches the controls."""
        session = self._context.require_session()

        if image_index < 0 or image_index >= len(session.images):
            raise IndexError(f"Image index {image_index} is out of range.")

        normalized_filter = self._validate_image_filter_arguments(
            filter_key=filter_key,
            confidence_threshold=confidence_threshold,
            class_id=class_id,
        )
        image_record = session.images[image_index]
        needs_annotations = (
            normalized_filter in self.FILTERS_REQUIRING_ANNOTATIONS
            or class_id is not None
        )
        loaded_for_filter = False

        if needs_annotations and not image_record.annotations_loaded:
            self._context.ensure_annotations_loaded(image_record)
            loaded_for_filter = True

        matches = self._image_matches_filter(
            image_record=image_record,
            filter_key=normalized_filter,
            confidence_threshold=confidence_threshold,
            class_id=class_id,
        )

        if loaded_for_filter and image_record is not session.current_image:
            image_record.unload_annotations()

        return matches

    def _validate_image_filter_arguments(
        self,
        filter_key: str,
        confidence_threshold: float,
        class_id: int | None,
    ) -> str:
        normalized_filter = filter_key.strip().casefold()

        if normalized_filter not in self.IMAGE_FILTER_KEYS:
            raise ValueError(f"Unknown image filter: {filter_key}")

        if not 0.0 <= confidence_threshold < 1.0:
            raise ValueError(
                "Confidence threshold must be at least 0 and less "
                "than 1."
            )

        session = self._context.require_session()

        if class_id is not None:
            validate_class_id(session, class_id)

        if normalized_filter == "missing_class" and class_id is None:
            raise ValueError(
                "Select a class before using Missing Selected Class."
            )

        return normalized_filter

    def _image_matches_filter(
        self,
        image_record: ImageRecord,
        filter_key: str,
        confidence_threshold: float,
        class_id: int | None,
    ) -> bool:
        annotations = image_record.annotations

        if filter_key == "missing_class":
            return not any(
                box.class_id == class_id for box in annotations
            )

        if filter_key == "all":
            primary_match = True
        elif filter_key == "unsaved":
            primary_match = not image_record.in_annotation_pool
        elif filter_key == "saved":
            primary_match = image_record.in_annotation_pool
        elif filter_key == "unsaved_changes":
            primary_match = image_record.is_dirty
        elif filter_key == "no_boxes":
            primary_match = not annotations
        elif filter_key == "single_box":
            primary_match = len(annotations) == 1
        elif filter_key == "multiple_boxes":
            primary_match = len(annotations) > 1
        elif filter_key == "manual":
            primary_match = any(
                box.source in {"manual", EDITED_SOURCE}
                for box in annotations
            )
        elif filter_key == "model_only":
            primary_match = bool(annotations) and all(
                box.source == MODEL_SOURCE
                for box in annotations
            )
        elif filter_key == "edited_model":
            primary_match = any(
                box.source == MODEL_EDITED_SOURCE
                for box in annotations
            )
        elif filter_key in {
            "confidence_below",
            "confidence_at_or_above",
        }:
            confidence_values = [
                box.confidence
                for box in annotations
                if (
                    box.confidence is not None
                    and box.source in {MODEL_SOURCE, MODEL_EDITED_SOURCE}
                    and (class_id is None or box.class_id == class_id)
                )
            ]

            if not confidence_values:
                return False

            image_confidence = min(confidence_values)

            if filter_key == "confidence_below":
                primary_match = image_confidence < confidence_threshold
            else:
                primary_match = image_confidence >= confidence_threshold
        else:
            return False

        if not primary_match:
            return False

        if class_id is None or filter_key == "no_boxes":
            return True

        return any(box.class_id == class_id for box in annotations)
