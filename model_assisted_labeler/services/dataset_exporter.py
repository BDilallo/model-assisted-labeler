from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from math import floor
from pathlib import Path
from typing import Callable
from uuid import uuid4

from model_assisted_labeler.models.annotation_session import ClassDefinition
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.annotation_store import YoloAnnotationStore
from model_assisted_labeler.services.image_service import ImageService
from model_assisted_labeler.services.session_repository import SessionRepository


ProgressCallback = Callable[[int, int, str], None]
CancellationCheck = Callable[[], bool]


class DatasetExportCancelled(RuntimeError):
    """Raised when the user cancels a dataset export."""


class DatasetExportDestinationExistsError(FileExistsError):
    """Raised when an export destination exists without replacement."""


class DatasetExportValidationError(ValueError):
    """Raised when the annotation pool cannot form a valid dataset."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        message = "Dataset export validation failed:\n\n" + "\n".join(
            f"- {error}" for error in self.errors
        )
        super().__init__(message)


@dataclass(frozen=True)
class DatasetExportRequest:
    """User-selected options for one YOLO dataset export."""

    dataset_name: str
    destination_parent: Path
    train_percent: int = 80
    validation_percent: int = 20
    test_percent: int = 0
    random_seed: int = 42
    replace_existing: bool = False

    def __post_init__(self) -> None:
        cleaned_name = self.dataset_name.strip()
        destination_parent = Path(self.destination_parent)

        if not cleaned_name:
            raise ValueError("Dataset name cannot be empty.")

        if cleaned_name in {".", ".."}:
            raise ValueError("Dataset name is invalid.")

        if any(character in cleaned_name for character in '<>:"/\\|?*'):
            raise ValueError(
                "Dataset name contains a character that cannot be used "
                "in a folder name."
            )

        percentages = (
            self.train_percent,
            self.validation_percent,
            self.test_percent,
        )

        if any(value < 0 or value > 100 for value in percentages):
            raise ValueError(
                "Dataset split percentages must be between 0 and 100."
            )

        if sum(percentages) != 100:
            raise ValueError(
                "Dataset split percentages must total 100%."
            )

        if self.train_percent == 0:
            raise ValueError("The training split must be greater than 0%.")

        if self.validation_percent == 0:
            raise ValueError(
                "The validation split must be greater than 0%."
            )

        object.__setattr__(self, "dataset_name", cleaned_name)
        object.__setattr__(self, "destination_parent", destination_parent)

    @property
    def export_directory(self) -> Path:
        return self.destination_parent / self.dataset_name

    @property
    def split_percentages(self) -> dict[str, int]:
        values = {
            "train": self.train_percent,
            "val": self.validation_percent,
        }

        if self.test_percent > 0:
            values["test"] = self.test_percent

        return values


@dataclass(frozen=True)
class DatasetExportResult:
    """Summary returned after a successful dataset export."""

    export_directory: Path
    total_images: int
    total_annotations: int
    split_counts: dict[str, int]
    class_mapping: dict[int, int]
    warnings: list[str]


@dataclass(frozen=True)
class _ExportRecord:
    image_path: Path
    label_path: Path
    annotations: list[BoundingBox]

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def class_ids(self) -> set[int]:
        return {box.class_id for box in self.annotations}


class DatasetExporter:
    """Build a portable Ultralytics YOLO dataset from a saved session pool."""

    def __init__(
        self,
        session_repository: SessionRepository,
        annotation_store: YoloAnnotationStore,
        image_service: ImageService | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._annotation_store = annotation_store
        self._image_service = image_service or ImageService()

    def export(
        self,
        definition: SessionDefinition,
        request: DatasetExportRequest,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> DatasetExportResult:
        """Validate, split, and export the annotation pool."""
        progress = progress_callback or self._ignore_progress
        is_cancelled = cancellation_check or self._never_cancelled

        final_directory = request.export_directory
        self._validate_destination(definition, final_directory)

        if final_directory.exists() and not request.replace_existing:
            raise DatasetExportDestinationExistsError(
                f"Dataset export already exists: {final_directory}"
            )

        image_paths, label_paths, discovery_errors = self._discover_pool(
            definition
        )

        if discovery_errors:
            raise DatasetExportValidationError(discovery_errors)

        total_records = len(image_paths)

        if total_records == 0:
            raise DatasetExportValidationError(
                ["The annotation pool does not contain any saved images."]
            )

        total_steps = (total_records * 2) + 3
        current_step = 0
        progress(current_step, total_steps, "Validating annotation pool...")

        records, validation_errors = self._load_and_validate_records(
            definition=definition,
            image_paths=image_paths,
            label_paths=label_paths,
            progress_callback=progress,
            cancellation_check=is_cancelled,
            current_step=current_step,
            total_steps=total_steps,
        )
        current_step += total_records

        if validation_errors:
            raise DatasetExportValidationError(validation_errors)

        class_mapping = self._build_class_mapping(definition.classes)
        split_records = self._split_records(records, request)
        warnings = self._build_warnings(
            records=records,
            split_records=split_records,
            classes=definition.classes,
        )

        self._ensure_not_cancelled(is_cancelled)
        progress(
            current_step,
            total_steps,
            "Preparing temporary export directory...",
        )

        request.destination_parent.mkdir(parents=True, exist_ok=True)
        temporary_directory = (
            request.destination_parent
            / f".{request.dataset_name}.exporting-{uuid4().hex}"
        )

        try:
            temporary_directory.mkdir(parents=False, exist_ok=False)

            for split_name, split_items in split_records.items():
                image_output_directory = (
                    temporary_directory / "images" / split_name
                )
                label_output_directory = (
                    temporary_directory / "labels" / split_name
                )
                image_output_directory.mkdir(parents=True, exist_ok=True)
                label_output_directory.mkdir(parents=True, exist_ok=True)

                for record in split_items:
                    self._ensure_not_cancelled(is_cancelled)
                    current_step += 1
                    progress(
                        current_step,
                        total_steps,
                        f"Exporting {split_name}: {record.filename}",
                    )
                    self._write_record(
                        record=record,
                        image_output_directory=image_output_directory,
                        label_output_directory=label_output_directory,
                        class_mapping=class_mapping,
                    )

            self._ensure_not_cancelled(is_cancelled)
            current_step += 1
            progress(
                current_step,
                total_steps,
                "Writing data.yaml...",
            )
            self._write_data_yaml(
                export_directory=temporary_directory,
                classes=definition.classes,
                class_mapping=class_mapping,
                include_test=(request.test_percent > 0),
            )

            current_step += 1
            progress(
                current_step,
                total_steps,
                "Writing export manifest...",
            )
            total_annotations = sum(
                len(record.annotations) for record in records
            )
            split_counts = {
                split_name: len(split_items)
                for split_name, split_items in split_records.items()
            }
            self._write_manifest(
                export_directory=temporary_directory,
                definition=definition,
                request=request,
                split_records=split_records,
                total_annotations=total_annotations,
                classes=definition.classes,
                class_mapping=class_mapping,
                warnings=warnings,
            )

            current_step += 1
            progress(
                current_step,
                total_steps,
                "Finalizing dataset export...",
            )
            self._commit_temporary_export(
                temporary_directory=temporary_directory,
                final_directory=final_directory,
                replace_existing=request.replace_existing,
            )

            progress(total_steps, total_steps, "Dataset export complete.")

            return DatasetExportResult(
                export_directory=final_directory,
                total_images=len(records),
                total_annotations=total_annotations,
                split_counts=split_counts,
                class_mapping=class_mapping,
                warnings=warnings,
            )

        except Exception:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

    def _discover_pool(
        self,
        definition: SessionDefinition,
    ) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
        image_directory = self._session_repository.annotated_images_directory(
            definition.session_directory
        )
        label_directory = self._session_repository.annotations_directory(
            definition.session_directory
        )
        errors: list[str] = []

        if not image_directory.is_dir():
            errors.append(
                f"Missing annotation-pool image directory: {image_directory}"
            )

        if not label_directory.is_dir():
            errors.append(
                f"Missing annotation-pool label directory: {label_directory}"
            )

        if errors:
            return {}, {}, errors

        image_paths, image_duplicates = self._paths_by_stem(
            path
            for path in image_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.casefold()
                in self._image_service.SUPPORTED_EXTENSIONS
            )
        )
        label_paths, label_duplicates = self._paths_by_stem(
            path
            for path in label_directory.glob("*.txt")
            if path.is_file()
        )

        for duplicate in image_duplicates:
            errors.append(
                "Multiple pooled images share the annotation stem "
                f"'{duplicate}'."
            )

        for duplicate in label_duplicates:
            errors.append(
                "Multiple pooled labels share the image stem "
                f"'{duplicate}'."
            )

        image_only = sorted(set(image_paths) - set(label_paths))
        label_only = sorted(set(label_paths) - set(image_paths))

        for stem in image_only:
            errors.append(
                f"Pooled image '{image_paths[stem].name}' has no matching "
                "annotation file."
            )

        for stem in label_only:
            errors.append(
                f"Annotation '{label_paths[stem].name}' has no matching "
                "pooled image."
            )

        matched_stems = set(image_paths) & set(label_paths)

        return (
            {stem: image_paths[stem] for stem in matched_stems},
            {stem: label_paths[stem] for stem in matched_stems},
            errors,
        )

    def _load_and_validate_records(
        self,
        definition: SessionDefinition,
        image_paths: dict[str, Path],
        label_paths: dict[str, Path],
        progress_callback: ProgressCallback,
        cancellation_check: CancellationCheck,
        current_step: int,
        total_steps: int,
    ) -> tuple[list[_ExportRecord], list[str]]:
        records: list[_ExportRecord] = []
        errors: list[str] = []
        valid_class_ids = {
            class_definition.class_id
            for class_definition in definition.classes
        }

        ordered_stems = sorted(
            image_paths,
            key=lambda stem: image_paths[stem].name.casefold(),
        )

        for offset, stem in enumerate(ordered_stems, start=1):
            self._ensure_not_cancelled(cancellation_check)
            image_path = image_paths[stem]
            label_path = label_paths[stem]
            progress_callback(
                current_step + offset,
                total_steps,
                f"Validating {image_path.name}",
            )

            try:
                width, height = self._image_service.get_dimensions(image_path)
                annotations = self._annotation_store.load(
                    label_path=label_path,
                    image_width=width,
                    image_height=height,
                )

                if not annotations:
                    raise ValueError("annotation file contains no boxes")

                undefined_ids = sorted(
                    {
                        box.class_id
                        for box in annotations
                        if box.class_id not in valid_class_ids
                    }
                )

                if undefined_ids:
                    joined_ids = ", ".join(str(value) for value in undefined_ids)
                    raise ValueError(
                        f"annotation uses undefined class ID(s): {joined_ids}"
                    )

                records.append(
                    _ExportRecord(
                        image_path=image_path,
                        label_path=label_path,
                        annotations=annotations,
                    )
                )

            except Exception as error:
                errors.append(f"{image_path.name}: {error}")

        return records, errors

    @staticmethod
    def _build_class_mapping(
        classes: list[ClassDefinition],
    ) -> dict[int, int]:
        if not classes:
            raise DatasetExportValidationError(
                ["The session does not define any annotation classes."]
            )

        return {
            class_definition.class_id: export_id
            for export_id, class_definition in enumerate(
                sorted(classes, key=lambda item: item.class_id)
            )
        }

    def _split_records(
        self,
        records: list[_ExportRecord],
        request: DatasetExportRequest,
    ) -> dict[str, list[_ExportRecord]]:
        shuffled_records = sorted(
            records,
            key=lambda record: record.filename.casefold(),
        )
        random.Random(request.random_seed).shuffle(shuffled_records)

        split_counts = self.calculate_split_counts(
            total_images=len(shuffled_records),
            split_percentages=request.split_percentages,
        )

        empty_splits = [
            name for name, count in split_counts.items() if count == 0
        ]

        if empty_splits:
            raise DatasetExportValidationError(
                [
                    "Not enough saved images to place at least one image "
                    "in every enabled split."
                ]
            )

        split_records: dict[str, list[_ExportRecord]] = {}
        start_index = 0

        for split_name in request.split_percentages:
            split_count = split_counts[split_name]
            end_index = start_index + split_count
            split_records[split_name] = shuffled_records[
                start_index:end_index
            ]
            start_index = end_index

        return split_records

    @staticmethod
    def calculate_split_counts(
        total_images: int,
        split_percentages: dict[str, int],
    ) -> dict[str, int]:
        """Convert percentages to exact counts using largest remainders."""
        if total_images < 0:
            raise ValueError("Total image count cannot be negative.")

        if sum(split_percentages.values()) != 100:
            raise ValueError("Split percentages must total 100%.")

        exact_counts = {
            name: (total_images * percentage) / 100.0
            for name, percentage in split_percentages.items()
        }
        counts = {
            name: floor(exact_count)
            for name, exact_count in exact_counts.items()
        }
        remaining = total_images - sum(counts.values())
        order = list(split_percentages)
        ranked_names = sorted(
            order,
            key=lambda name: (
                exact_counts[name] - counts[name],
                -order.index(name),
            ),
            reverse=True,
        )

        for index in range(remaining):
            counts[ranked_names[index % len(ranked_names)]] += 1

        enabled_names = [
            name
            for name, percentage in split_percentages.items()
            if percentage > 0
        ]

        if total_images >= len(enabled_names):
            for empty_name in [
                name for name in enabled_names if counts[name] == 0
            ]:
                donor_name = max(
                    (
                        name
                        for name in enabled_names
                        if counts[name] > 1
                    ),
                    key=lambda name: counts[name],
                    default=None,
                )

                if donor_name is None:
                    break

                counts[donor_name] -= 1
                counts[empty_name] += 1

        return counts

    @staticmethod
    def _build_warnings(
        records: list[_ExportRecord],
        split_records: dict[str, list[_ExportRecord]],
        classes: list[ClassDefinition],
    ) -> list[str]:
        warnings: list[str] = []
        all_used_ids = set().union(
            *(record.class_ids for record in records)
        )

        for class_definition in classes:
            if class_definition.class_id not in all_used_ids:
                warnings.append(
                    f"Class '{class_definition.name}' has no annotations "
                    "in the exported dataset."
                )

        used_classes = [
            class_definition
            for class_definition in classes
            if class_definition.class_id in all_used_ids
        ]

        for split_name, split_items in split_records.items():
            if not split_items:
                warnings.append(
                    f"The {split_name} split contains no images."
                )
                continue

            split_class_ids = set().union(
                *(record.class_ids for record in split_items)
            )

            for class_definition in used_classes:
                if class_definition.class_id not in split_class_ids:
                    warnings.append(
                        f"Class '{class_definition.name}' has no images "
                        f"in the {split_name} split."
                    )

        return warnings

    def _write_record(
        self,
        record: _ExportRecord,
        image_output_directory: Path,
        label_output_directory: Path,
        class_mapping: dict[int, int],
    ) -> None:
        output_image_path = image_output_directory / record.image_path.name
        output_label_path = (
            label_output_directory
            / record.image_path.with_suffix(".txt").name
        )
        shutil.copy2(record.image_path, output_image_path)

        remapped_annotations = [
            replace(box, class_id=class_mapping[box.class_id])
            for box in record.annotations
        ]
        width, height = self._image_service.get_dimensions(record.image_path)
        self._annotation_store.save(
            label_path=output_label_path,
            annotations=remapped_annotations,
            image_width=width,
            image_height=height,
        )

    @staticmethod
    def _write_data_yaml(
        export_directory: Path,
        classes: list[ClassDefinition],
        class_mapping: dict[int, int],
        include_test: bool,
    ) -> None:
        # Omitting ``path`` makes current Ultralytics versions resolve
        # train/val/test relative to this YAML file, so the exported
        # dataset remains portable when moved to another directory.
        lines = [
            "train: images/train",
            "val: images/val",
        ]

        if include_test:
            lines.append("test: images/test")

        lines.extend(["", "names:"])

        for class_definition in sorted(
            classes,
            key=lambda item: class_mapping[item.class_id],
        ):
            export_id = class_mapping[class_definition.class_id]
            yaml_name = json.dumps(class_definition.name, ensure_ascii=False)
            lines.append(f"  {export_id}: {yaml_name}")

        (export_directory / "data.yaml").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_manifest(
        export_directory: Path,
        definition: SessionDefinition,
        request: DatasetExportRequest,
        split_records: dict[str, list[_ExportRecord]],
        total_annotations: int,
        classes: list[ClassDefinition],
        class_mapping: dict[int, int],
        warnings: list[str],
    ) -> None:
        payload = {
            "dataset_name": request.dataset_name,
            "source_session": definition.name,
            "exported_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "format": "yolo_detection",
            "random_seed": request.random_seed,
            "split_percentages": request.split_percentages,
            "split_counts": {
                split_name: len(split_items)
                for split_name, split_items in split_records.items()
            },
            "total_images": sum(
                len(split_items) for split_items in split_records.values()
            ),
            "total_annotations": total_annotations,
            "class_mapping": [
                {
                    "session_id": class_definition.class_id,
                    "export_id": class_mapping[class_definition.class_id],
                    "name": class_definition.name,
                }
                for class_definition in sorted(
                    classes,
                    key=lambda item: class_mapping[item.class_id],
                )
            ],
            "files": {
                split_name: [record.filename for record in split_items]
                for split_name, split_items in split_records.items()
            },
            "warnings": warnings,
        }

        (export_directory / "export_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _commit_temporary_export(
        temporary_directory: Path,
        final_directory: Path,
        replace_existing: bool,
    ) -> None:
        backup_directory: Path | None = None

        if final_directory.exists():
            if not replace_existing:
                raise DatasetExportDestinationExistsError(
                    f"Dataset export already exists: {final_directory}"
                )

            if not final_directory.is_dir():
                raise ValueError(
                    f"Export destination is not a directory: {final_directory}"
                )

            backup_directory = final_directory.with_name(
                f".{final_directory.name}.backup-{uuid4().hex}"
            )
            final_directory.replace(backup_directory)

        try:
            temporary_directory.replace(final_directory)
        except Exception:
            if backup_directory is not None and backup_directory.exists():
                backup_directory.replace(final_directory)
            raise
        else:
            if backup_directory is not None and backup_directory.exists():
                shutil.rmtree(backup_directory, ignore_errors=True)

    @staticmethod
    def _validate_destination(
        definition: SessionDefinition,
        final_directory: Path,
    ) -> None:
        destination = final_directory.resolve(strict=False)
        protected_paths = [
            definition.image_directory.resolve(),
            definition.session_directory.resolve(),
        ]

        for protected_path in protected_paths:
            if (
                destination == protected_path
                or destination in protected_path.parents
                or protected_path in destination.parents
            ):
                raise ValueError(
                    "The export destination cannot overlap the source image "
                    "directory or the active session directory."
                )

    @staticmethod
    def _paths_by_stem(
        paths,
    ) -> tuple[dict[str, Path], set[str]]:
        values: dict[str, Path] = {}
        duplicates: set[str] = set()

        for path in paths:
            normalized_stem = path.stem.casefold()

            if normalized_stem in values:
                duplicates.add(path.stem)
            else:
                values[normalized_stem] = path

        return values, duplicates

    @staticmethod
    def _ensure_not_cancelled(
        cancellation_check: CancellationCheck,
    ) -> None:
        if cancellation_check():
            raise DatasetExportCancelled("Dataset export was cancelled.")

    @staticmethod
    def _ignore_progress(current: int, total: int, message: str) -> None:
        del current, total, message

    @staticmethod
    def _never_cancelled() -> bool:
        return False
