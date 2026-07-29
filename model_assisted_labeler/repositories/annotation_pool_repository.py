import json
import shutil
from pathlib import Path

from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.repositories.atomic_file_io import (
    atomic_write_text,
)
from model_assisted_labeler.repositories.session_document_repository import (
    SessionDocumentRepository,
)
from model_assisted_labeler.repositories.session_paths import (
    annotated_image_path_for,
    annotated_images_directory,
    annotation_metadata_directory,
    annotation_metadata_path_for,
    annotation_path_for,
    annotations_directory,
    count_pooled_images,
    validate_source_image_path,
)


class AnnotationPoolRepository:
    """
    Owns the pooled (saved) annotations for a session.

    An image "enters the pool" once its YOLO label file, its copied
    source image, and its confidence/source metadata sidecar have all
    been written under the session directory.
    """

    def __init__(
        self,
        annotation_store: YoloAnnotationStore,
        document_repository: SessionDocumentRepository,
    ) -> None:
        self._annotation_store = annotation_store
        self._document_repository = document_repository

    def load_annotations(
        self,
        definition: SessionDefinition,
        image_record: ImageRecord,
    ) -> list[BoundingBox]:
        annotation_path = annotation_path_for(
            definition,
            image_record.image_path,
        )
        annotations = self._annotation_store.load(
            label_path=annotation_path,
            image_width=image_record.width,
            image_height=image_record.height,
        )
        metadata_path = annotation_metadata_path_for(
            definition,
            image_record.image_path,
        )
        self._apply_annotation_metadata(metadata_path, annotations)
        return annotations

    def image_is_in_pool(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> bool:
        annotation_path = annotation_path_for(
            definition,
            image_path,
        )
        copied_image_path = annotated_image_path_for(
            definition,
            image_path,
        )

        return annotation_path.is_file() and copied_image_path.is_file()

    def save_image_to_pool(
        self,
        definition: SessionDefinition,
        image_record: ImageRecord,
        refresh_session_info: bool = True,
    ) -> None:
        if not image_record.annotations:
            raise ValueError(
                "An image must contain at least one box before it can "
                "be saved."
            )

        validate_source_image_path(
            definition,
            image_record.image_path,
        )

        if not image_record.image_path.is_file():
            raise FileNotFoundError(
                f"Source image does not exist: {image_record.image_path}"
            )

        annotation_path = annotation_path_for(
            definition,
            image_record.image_path,
        )
        copied_image_path = annotated_image_path_for(
            definition,
            image_record.image_path,
        )

        self._annotation_store.save(
            label_path=annotation_path,
            annotations=image_record.annotations,
            image_width=image_record.width,
            image_height=image_record.height,
        )
        self._save_annotation_metadata(
            annotation_metadata_path_for(
                definition,
                image_record.image_path,
            ),
            image_record.annotations,
        )

        temporary_image = copied_image_path.with_suffix(
            copied_image_path.suffix + ".tmp"
        )

        try:
            shutil.copy2(image_record.image_path, temporary_image)
            temporary_image.replace(copied_image_path)
        finally:
            if temporary_image.exists():
                temporary_image.unlink()

        if refresh_session_info:
            definition.total_images_annotated = count_pooled_images(
                definition.session_directory
            )
            self._document_repository.save_session_info(definition)

    def remove_image_from_pool(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> None:
        annotation_path = annotation_path_for(
            definition,
            image_path,
        )
        copied_image_path = annotated_image_path_for(
            definition,
            image_path,
        )
        metadata_path = annotation_metadata_path_for(
            definition,
            image_path,
        )

        if annotation_path.exists():
            annotation_path.unlink()

        if copied_image_path.exists():
            copied_image_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        definition.total_images_annotated = count_pooled_images(
            definition.session_directory
        )
        self._document_repository.save_session_info(definition)

    def clear_pool(self, definition: SessionDefinition) -> None:
        """Delete every pooled label, copied image, and metadata file."""
        for directory in (
            annotations_directory(definition.session_directory),
            annotated_images_directory(definition.session_directory),
            annotation_metadata_directory(definition.session_directory),
        ):
            if not directory.is_dir():
                continue

            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()

        definition.total_images_annotated = 0
        self._document_repository.save_session_info(definition)

    def class_usage_filenames(
        self,
        definition: SessionDefinition,
        class_id: int,
    ) -> list[str]:
        matches: list[str] = []
        annotation_directory = annotations_directory(
            definition.session_directory
        )

        for annotation_path in sorted(annotation_directory.glob("*.txt")):
            if self._annotation_file_contains_class(
                annotation_path,
                class_id,
            ):
                matches.append(annotation_path.stem)

        return matches

    def remove_class_from_pool_annotations(
        self,
        definition: SessionDefinition,
        class_id: int,
        delete_referenced_images: bool,
    ) -> set[str]:
        """
        Remove one class from persisted annotations.

        When ``delete_referenced_images`` is true, every pooled image
        containing the class is removed entirely. Otherwise only rows
        using the class are removed; files left empty are also removed
        from the annotation pool.
        """
        affected_stems: set[str] = set()
        annotation_directory = annotations_directory(
            definition.session_directory
        )

        for annotation_path in sorted(annotation_directory.glob("*.txt")):
            original_lines = self._read_annotation_lines(annotation_path)
            retained_indexes = [
                index
                for index, line in enumerate(original_lines)
                if self._line_class_id(line) != class_id
            ]

            if len(retained_indexes) == len(original_lines):
                continue

            affected_stems.add(annotation_path.stem)
            metadata_path = (
                annotation_metadata_directory(
                    definition.session_directory
                )
                / f"{annotation_path.stem}.json"
            )

            if delete_referenced_images:
                annotation_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                self._remove_copied_image_by_stem(
                    definition,
                    annotation_path.stem,
                )
                continue

            remaining_lines = [
                original_lines[index] for index in retained_indexes
            ]

            if remaining_lines:
                atomic_write_text(
                    annotation_path,
                    "\n".join(remaining_lines) + "\n",
                )
                self._retain_annotation_metadata_rows(
                    metadata_path,
                    retained_indexes,
                )
            else:
                annotation_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                self._remove_copied_image_by_stem(
                    definition,
                    annotation_path.stem,
                )

        definition.total_images_annotated = count_pooled_images(
            definition.session_directory
        )
        self._document_repository.save_session_info(definition)
        return affected_stems

    @staticmethod
    def _read_annotation_lines(annotation_path: Path) -> list[str]:
        return [
            line.strip()
            for line in annotation_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    def _annotation_file_contains_class(
        self,
        annotation_path: Path,
        class_id: int,
    ) -> bool:
        return any(
            self._line_class_id(line) == class_id
            for line in self._read_annotation_lines(annotation_path)
        )

    @staticmethod
    def _line_class_id(line: str) -> int | None:
        values = line.split()

        if not values:
            return None

        try:
            return int(values[0])
        except ValueError:
            return None

    def _save_annotation_metadata(
        self,
        metadata_path: Path,
        annotations: list[BoundingBox],
    ) -> None:
        payload = {
            "version": 1,
            "annotations": [
                {
                    "confidence": annotation.confidence,
                    "source": annotation.source,
                }
                for annotation in annotations
            ],
        }
        atomic_write_text(
            metadata_path,
            json.dumps(payload, indent=2) + "\n",
        )

    @staticmethod
    def _apply_annotation_metadata(
        metadata_path: Path,
        annotations: list[BoundingBox],
    ) -> None:
        if not metadata_path.is_file():
            return

        try:
            payload = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return

        rows = payload.get("annotations")

        if not isinstance(rows, list) or len(rows) != len(annotations):
            return

        for annotation, row in zip(annotations, rows):
            if not isinstance(row, dict):
                continue

            confidence = row.get("confidence")

            if confidence is None:
                annotation.confidence = None
            elif isinstance(confidence, (int, float)):
                normalized_confidence = float(confidence)

                if 0.0 <= normalized_confidence <= 1.0:
                    annotation.confidence = normalized_confidence

            source = row.get("source")

            if isinstance(source, str) and source.strip():
                annotation.source = source.strip()

    def _retain_annotation_metadata_rows(
        self,
        metadata_path: Path,
        retained_indexes: list[int],
    ) -> None:
        if not metadata_path.is_file():
            return

        try:
            payload = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            metadata_path.unlink(missing_ok=True)
            return

        rows = payload.get("annotations")

        if not isinstance(rows, list):
            metadata_path.unlink(missing_ok=True)
            return

        retained_rows = [
            rows[index]
            for index in retained_indexes
            if 0 <= index < len(rows)
        ]

        if len(retained_rows) != len(retained_indexes):
            metadata_path.unlink(missing_ok=True)
            return

        payload["annotations"] = retained_rows
        atomic_write_text(
            metadata_path,
            json.dumps(payload, indent=2) + "\n",
        )

    def _remove_copied_image_by_stem(
        self,
        definition: SessionDefinition,
        stem: str,
    ) -> None:
        image_directory = annotated_images_directory(
            definition.session_directory
        )
        normalized_stem = stem.casefold()

        for candidate in image_directory.iterdir():
            if (
                candidate.is_file()
                and candidate.stem.casefold() == normalized_stem
            ):
                candidate.unlink()
