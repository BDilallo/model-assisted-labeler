from dataclasses import replace

from model_assisted_labeler.models.bounding_box import BoundingBox


class AnnotationEditingMixin:
    """Add, edit, and persist per-image annotations."""

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
