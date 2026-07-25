import random
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from model_assisted_labeler.models.class_definition import ClassDefinition
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories import session_paths

ProgressCallback = Callable[[int, int, str], None]
CancellationCallback = Callable[[], bool]


class DatasetExportCancelled(Exception):
    """Raised when the caller reports cancellation mid-export."""


@dataclass
class DatasetExportSettings:
    """User-selected options for a dataset export."""

    output_directory: Path
    dataset_folder_name: str
    validation_is_percent: bool
    validation_amount: float
    shuffle: bool = True
    seed: int | None = None
    remap_class_ids: bool = True


@dataclass(frozen=True)
class DatasetExportResult:
    """Summary of one completed dataset export."""

    output_path: Path
    train_count: int
    val_count: int
    class_count: int


class DatasetExportService:
    """Build a ready-to-train YOLO dataset folder from pooled annotations.

    Only images already saved to the session's annotation pool are
    exported. Source images are never read from or modified; the
    session's own pooled image copies and label files are used instead.
    """

    IMAGES_DIRECTORY = "images"
    LABELS_DIRECTORY = "labels"
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "val"

    def compute_split_counts(
        self,
        total_images: int,
        settings: DatasetExportSettings,
    ) -> tuple[int, int]:
        """Return ``(train_count, val_count)`` for preview and export."""
        if total_images <= 0:
            return 0, 0

        if settings.validation_is_percent:
            val_count = round(
                total_images * settings.validation_amount / 100.0
            )
        else:
            val_count = round(settings.validation_amount)

        val_count = max(0, min(val_count, total_images))
        return total_images - val_count, val_count

    def export(
        self,
        definition: SessionDefinition,
        pooled_images: list[ImageRecord],
        classes: list[ClassDefinition],
        settings: DatasetExportSettings,
        progress_callback: ProgressCallback | None = None,
        cancellation_requested: CancellationCallback | None = None,
    ) -> DatasetExportResult:
        if not pooled_images:
            raise ValueError("There are no saved images to export.")

        if not classes:
            raise ValueError("The session has no classes to export.")

        folder_name = settings.dataset_folder_name.strip()

        if not folder_name:
            raise ValueError("Enter a name for the dataset folder.")

        if not settings.output_directory.is_dir():
            raise ValueError(
                "The selected output directory does not exist."
            )

        final_path = settings.output_directory / folder_name

        if final_path.exists() and any(final_path.iterdir()):
            raise FileExistsError(
                f"'{final_path}' already exists and is not empty."
            )

        ordered_images = list(pooled_images)

        if settings.shuffle:
            random.Random(settings.seed).shuffle(ordered_images)

        _, val_count = self.compute_split_counts(
            len(ordered_images),
            settings,
        )
        val_images = ordered_images[:val_count]
        train_images = ordered_images[val_count:]

        class_id_map = self._build_class_id_map(
            classes,
            settings.remap_class_ids,
        )
        ordered_names = self._ordered_class_names(
            classes,
            class_id_map,
            settings.remap_class_ids,
        )

        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f"{folder_name}.",
                dir=str(settings.output_directory),
            )
        )

        try:
            total = len(train_images) + len(val_images)
            completed = 0

            for split_name, split_images in (
                (self.TRAIN_SPLIT, train_images),
                (self.VAL_SPLIT, val_images),
            ):
                images_directory = (
                    temporary_root / self.IMAGES_DIRECTORY / split_name
                )
                labels_directory = (
                    temporary_root / self.LABELS_DIRECTORY / split_name
                )
                images_directory.mkdir(parents=True, exist_ok=True)
                labels_directory.mkdir(parents=True, exist_ok=True)

                for image_record in split_images:
                    if (
                        cancellation_requested is not None
                        and cancellation_requested()
                    ):
                        raise DatasetExportCancelled()

                    self._export_one_image(
                        definition=definition,
                        image_record=image_record,
                        images_directory=images_directory,
                        labels_directory=labels_directory,
                        class_id_map=class_id_map,
                    )
                    completed += 1

                    if progress_callback is not None:
                        progress_callback(
                            completed,
                            total,
                            image_record.filename,
                        )

            self._write_classes_file(temporary_root, ordered_names)
            self._write_data_yaml(
                temporary_root,
                final_path,
                ordered_names,
            )

            if final_path.exists():
                final_path.rmdir()

            shutil.move(str(temporary_root), str(final_path))

        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise

        return DatasetExportResult(
            output_path=final_path,
            train_count=len(train_images),
            val_count=len(val_images),
            class_count=len(ordered_names),
        )

    def _export_one_image(
        self,
        definition: SessionDefinition,
        image_record: ImageRecord,
        images_directory: Path,
        labels_directory: Path,
        class_id_map: dict[int, int],
    ) -> None:
        pooled_image_path = session_paths.annotated_image_path_for(
            definition,
            image_record.image_path,
        )

        if not pooled_image_path.is_file():
            raise FileNotFoundError(
                "Missing pooled image copy for "
                f"'{image_record.filename}'."
            )

        shutil.copy2(
            pooled_image_path,
            images_directory / pooled_image_path.name,
        )

        label_lines = self._remapped_label_lines(
            image_record.label_path,
            class_id_map,
        )
        label_destination = (
            labels_directory / f"{pooled_image_path.stem}.txt"
        )
        label_text = "\n".join(label_lines)

        if label_lines:
            label_text += "\n"

        label_destination.write_text(label_text, encoding="utf-8")

    @staticmethod
    def _remapped_label_lines(
        label_path: Path,
        class_id_map: dict[int, int],
    ) -> list[str]:
        if not label_path.is_file():
            return []

        remapped_lines: list[str] = []

        for raw_line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            stripped_line = raw_line.strip()

            if not stripped_line:
                continue

            values = stripped_line.split()
            original_class_id = int(values[0])
            values[0] = str(
                class_id_map.get(original_class_id, original_class_id)
            )
            remapped_lines.append(" ".join(values))

        return remapped_lines

    @staticmethod
    def _build_class_id_map(
        classes: list[ClassDefinition],
        remap: bool,
    ) -> dict[int, int]:
        if not remap:
            return {
                class_definition.class_id: class_definition.class_id
                for class_definition in classes
            }

        sorted_classes = sorted(classes, key=lambda item: item.class_id)
        return {
            class_definition.class_id: new_id
            for new_id, class_definition in enumerate(sorted_classes)
        }

    @staticmethod
    def _ordered_class_names(
        classes: list[ClassDefinition],
        class_id_map: dict[int, int],
        remap: bool,
    ) -> list[str]:
        if remap:
            pairs = sorted(
                (
                    (class_id_map[class_definition.class_id], class_definition.name)
                    for class_definition in classes
                ),
                key=lambda pair: pair[0],
            )
            return [name for _, name in pairs]

        # Session class IDs may have gaps (e.g. after a class was
        # deleted). YOLO requires label class IDs to index directly
        # into ``names``, so unused indexes are padded with placeholder
        # names rather than compacted.
        highest_id = max(
            class_definition.class_id for class_definition in classes
        )
        names = [f"unused_{index}" for index in range(highest_id + 1)]

        for class_definition in classes:
            names[class_definition.class_id] = class_definition.name

        return names

    @staticmethod
    def _write_classes_file(
        root: Path,
        class_names: list[str],
    ) -> None:
        (root / "classes.txt").write_text(
            "\n".join(class_names) + "\n",
            encoding="utf-8",
        )

    def _write_data_yaml(
        self,
        temporary_root: Path,
        final_path: Path,
        class_names: list[str],
    ) -> None:
        images_root = final_path / self.IMAGES_DIRECTORY
        lines = [
            f"train: {(images_root / self.TRAIN_SPLIT).as_posix()}",
            f"val: {(images_root / self.VAL_SPLIT).as_posix()}",
            f"nc: {len(class_names)}",
            "names:",
        ]

        for class_name in class_names:
            lines.append(f"  - {self._yaml_escape(class_name)}")

        (temporary_root / "data.yaml").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _yaml_escape(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"
