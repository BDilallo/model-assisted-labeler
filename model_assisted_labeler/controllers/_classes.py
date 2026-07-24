from model_assisted_labeler.models.annotation_session import ClassDefinition


class ClassManagementMixin:
    """Add, inspect, and delete session classes."""

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
