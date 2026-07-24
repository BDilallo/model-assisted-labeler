from pathlib import Path

from model_assisted_labeler.models.annotation_session import (
    AnnotationSession,
    ClassDefinition,
)


class ModelManagementMixin:
    """Load detection models and validate their class mappings."""

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
