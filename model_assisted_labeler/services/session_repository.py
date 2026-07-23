import json
import shutil
from pathlib import Path

from model_assisted_labeler.models.annotation_session import ClassDefinition
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.annotation_store import YoloAnnotationStore
from model_assisted_labeler.services.image_service import ImageService


class SessionAlreadyExistsError(ValueError):
    """Raised when a session name is already in use."""


class SessionRepository:
    """
    Owns all persistent session reads and writes.

    Source image directories are treated as read-only. Every mutation
    performed by this class is constrained to the program-owned
    ``Open Sessions`` directory.
    """

    OPEN_SESSIONS_DIRECTORY = "Open Sessions"
    SESSION_INFO_FILENAME = "Session Info.txt"
    CLASSES_FILENAME = "Classes.txt"
    ANNOTATED_IMAGES_DIRECTORY = "Annotated Images"
    ANNOTATIONS_DIRECTORY = "Annotations"
    ANNOTATION_METADATA_DIRECTORY = "Annotation Metadata"

    def __init__(
        self,
        workspace_root: Path,
        annotation_store: YoloAnnotationStore,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.open_sessions_directory = (
            self.workspace_root / self.OPEN_SESSIONS_DIRECTORY
        )
        self.annotation_store = annotation_store

        self.open_sessions_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_session(
        self,
        name: str,
        image_directory: Path,
        model_paths: list[Path],
        classes: list[ClassDefinition],
    ) -> SessionDefinition:
        cleaned_name = self._validate_session_name(name)
        image_directory = Path(image_directory).resolve()
        model_paths = [Path(path).resolve() for path in model_paths]

        if not image_directory.is_dir():
            raise ValueError(
                f"Image directory does not exist: {image_directory}"
            )

        self._validate_model_paths(model_paths)
        self._validate_classes(classes)
        self._validate_source_workspace_separation(image_directory)
        self._validate_flat_image_names(image_directory)

        if self._session_name_exists(cleaned_name):
            raise SessionAlreadyExistsError("Session already exists")

        session_directory = self.open_sessions_directory / cleaned_name

        try:
            session_directory.mkdir(parents=False, exist_ok=False)
            self.annotated_images_directory(session_directory).mkdir()
            self.annotations_directory(session_directory).mkdir()
            self.annotation_metadata_directory(session_directory).mkdir()

            next_class_id = (
                max((item.class_id for item in classes), default=-1) + 1
            )

            definition = SessionDefinition(
                name=cleaned_name,
                session_directory=session_directory,
                image_directory=image_directory,
                model_paths=model_paths,
                classes=sorted(classes, key=lambda item: item.class_id),
                last_image_loaded=None,
                total_images_annotated=0,
                next_class_id=next_class_id,
            )

            self.save_classes(definition)
            self.save_session_info(definition)
            return definition

        except Exception:
            if session_directory.exists():
                shutil.rmtree(session_directory, ignore_errors=True)
            raise

    def list_sessions(self) -> list[SessionDefinition]:
        definitions: list[SessionDefinition] = []

        for candidate in self.open_sessions_directory.iterdir():
            if not candidate.is_dir():
                continue

            try:
                definitions.append(self.load_session(candidate))
            except (OSError, ValueError):
                continue

        definitions.sort(key=lambda item: item.name.casefold())
        return definitions

    def load_session(
        self,
        session: str | Path,
    ) -> SessionDefinition:
        if isinstance(session, Path):
            session_directory = session
        else:
            session_directory = self.open_sessions_directory / session

        session_directory = session_directory.resolve()
        self._assert_session_directory(session_directory)

        if not session_directory.is_dir():
            raise FileNotFoundError(
                f"Session does not exist: {session_directory.name}"
            )

        info = self._read_session_info(session_directory)
        classes = self._read_classes(session_directory)

        name = info.get("Session Name", session_directory.name).strip()
        image_source = info.get("Image Source", "").strip()

        if not image_source:
            raise ValueError(
                f"Session '{name}' does not contain an image source."
            )

        model_paths = self._parse_model_paths(info)
        last_image_loaded = (
            info.get("Last Image Loaded", "").strip() or None
        )

        try:
            next_class_id = int(
                info.get(
                    "Next Class ID",
                    str(
                        max(
                            (item.class_id for item in classes),
                            default=-1,
                        )
                        + 1
                    ),
                )
            )
        except ValueError:
            next_class_id = (
                max((item.class_id for item in classes), default=-1) + 1
            )

        definition = SessionDefinition(
            name=name,
            session_directory=session_directory,
            image_directory=Path(image_source),
            model_paths=model_paths,
            classes=classes,
            last_image_loaded=last_image_loaded,
            total_images_annotated=self._count_pooled_images(
                session_directory
            ),
            next_class_id=max(next_class_id, 0),
        )

        self._ensure_session_structure(definition)
        self.save_session_info(definition)
        return definition

    def delete_session(self, definition: SessionDefinition) -> None:
        session_directory = definition.session_directory.resolve()
        self._assert_session_directory(session_directory)

        if session_directory.exists():
            shutil.rmtree(session_directory)

    def update_paths(
        self,
        definition: SessionDefinition,
        image_directory: Path,
        model_paths: list[Path],
    ) -> None:
        image_directory = Path(image_directory).resolve()
        model_paths = [Path(path).resolve() for path in model_paths]

        if not image_directory.is_dir():
            raise ValueError(
                f"Image directory does not exist: {image_directory}"
            )

        self._validate_model_paths(model_paths)
        self._validate_source_workspace_separation(image_directory)
        self._validate_flat_image_names(image_directory)

        definition.image_directory = image_directory
        definition.model_paths = model_paths
        self.save_session_info(definition)

    def update_last_image(
        self,
        definition: SessionDefinition,
        filename: str | None,
    ) -> None:
        definition.last_image_loaded = filename
        self.save_session_info(definition)

    def save_session_info(
        self,
        definition: SessionDefinition,
    ) -> None:
        self._ensure_session_structure(definition)
        definition.total_images_annotated = self._count_pooled_images(
            definition.session_directory
        )

        model_values = [
            str(path.resolve()) for path in definition.model_paths
        ]

        lines = [
            f"Session Name={definition.name}",
            f"Image Source={definition.image_directory.resolve()}",
            f"Model Sources={json.dumps(model_values)}",
            (
                "Last Image Loaded="
                f"{definition.last_image_loaded or ''}"
            ),
            (
                "Total Images Annotated="
                f"{definition.total_images_annotated}"
            ),
            f"Next Class ID={definition.next_class_id}",
        ]

        self._atomic_write_text(
            definition.session_directory / self.SESSION_INFO_FILENAME,
            "\n".join(lines) + "\n",
        )

    def save_classes(self, definition: SessionDefinition) -> None:
        self._validate_classes(definition.classes)

        lines = [
            f"{item.class_id}\t{item.name}"
            for item in sorted(
                definition.classes,
                key=lambda class_item: class_item.class_id,
            )
        ]

        output = "\n".join(lines)

        if lines:
            output += "\n"

        self._atomic_write_text(
            definition.session_directory / self.CLASSES_FILENAME,
            output,
        )
        self.save_session_info(definition)

    def annotation_path_for(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> Path:
        self._validate_source_image_path(definition, image_path)
        return (
            self.annotations_directory(definition.session_directory)
            / Path(image_path).with_suffix(".txt").name
        )

    def annotated_image_path_for(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> Path:
        self._validate_source_image_path(definition, image_path)
        return (
            self.annotated_images_directory(
                definition.session_directory
            )
            / Path(image_path).name
        )

    def annotation_metadata_path_for(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> Path:
        self._validate_source_image_path(definition, image_path)
        return (
            self.annotation_metadata_directory(
                definition.session_directory
            )
            / f"{Path(image_path).stem}.json"
        )

    def load_annotations(
        self,
        definition: SessionDefinition,
        image_record: ImageRecord,
    ) -> list:
        annotation_path = self.annotation_path_for(
            definition,
            image_record.image_path,
        )
        annotations = self.annotation_store.load(
            label_path=annotation_path,
            image_width=image_record.width,
            image_height=image_record.height,
        )
        metadata_path = self.annotation_metadata_path_for(
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
        annotation_path = self.annotation_path_for(
            definition,
            image_path,
        )
        copied_image_path = self.annotated_image_path_for(
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

        self._validate_source_image_path(
            definition,
            image_record.image_path,
        )

        if not image_record.image_path.is_file():
            raise FileNotFoundError(
                f"Source image does not exist: {image_record.image_path}"
            )

        annotation_path = self.annotation_path_for(
            definition,
            image_record.image_path,
        )
        copied_image_path = self.annotated_image_path_for(
            definition,
            image_record.image_path,
        )

        self.annotation_store.save(
            label_path=annotation_path,
            annotations=image_record.annotations,
            image_width=image_record.width,
            image_height=image_record.height,
        )
        self._save_annotation_metadata(
            self.annotation_metadata_path_for(
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
            definition.total_images_annotated = self._count_pooled_images(
                definition.session_directory
            )
            self.save_session_info(definition)

    def remove_image_from_pool(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> None:
        annotation_path = self.annotation_path_for(
            definition,
            image_path,
        )
        copied_image_path = self.annotated_image_path_for(
            definition,
            image_path,
        )
        metadata_path = self.annotation_metadata_path_for(
            definition,
            image_path,
        )

        if annotation_path.exists():
            annotation_path.unlink()

        if copied_image_path.exists():
            copied_image_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        definition.total_images_annotated = self._count_pooled_images(
            definition.session_directory
        )
        self.save_session_info(definition)

    def class_usage_filenames(
        self,
        definition: SessionDefinition,
        class_id: int,
    ) -> list[str]:
        matches: list[str] = []
        annotation_directory = self.annotations_directory(
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
        annotation_directory = self.annotations_directory(
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
                self.annotation_metadata_directory(
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
                self._atomic_write_text(
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

        definition.total_images_annotated = self._count_pooled_images(
            definition.session_directory
        )
        self.save_session_info(definition)
        return affected_stems

    def annotations_directory(self, session_directory: Path) -> Path:
        return Path(session_directory) / self.ANNOTATIONS_DIRECTORY

    def annotated_images_directory(
        self,
        session_directory: Path,
    ) -> Path:
        return Path(session_directory) / self.ANNOTATED_IMAGES_DIRECTORY

    def annotation_metadata_directory(
        self,
        session_directory: Path,
    ) -> Path:
        return (
            Path(session_directory)
            / self.ANNOTATION_METADATA_DIRECTORY
        )

    def _ensure_session_structure(
        self,
        definition: SessionDefinition,
    ) -> None:
        self._assert_session_directory(
            definition.session_directory.resolve()
        )
        definition.session_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.annotated_images_directory(
            definition.session_directory
        ).mkdir(parents=True, exist_ok=True)
        self.annotations_directory(
            definition.session_directory
        ).mkdir(parents=True, exist_ok=True)
        self.annotation_metadata_directory(
            definition.session_directory
        ).mkdir(parents=True, exist_ok=True)

    def _read_session_info(
        self,
        session_directory: Path,
    ) -> dict[str, str]:
        info_path = session_directory / self.SESSION_INFO_FILENAME

        if not info_path.is_file():
            raise ValueError(
                f"Missing {self.SESSION_INFO_FILENAME} in "
                f"'{session_directory.name}'."
            )

        values: dict[str, str] = {}

        for raw_line in info_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not raw_line.strip() or "=" not in raw_line:
                continue

            key, value = raw_line.split("=", 1)
            values[key.strip()] = value.strip()

        return values

    def _read_classes(
        self,
        session_directory: Path,
    ) -> list[ClassDefinition]:
        classes_path = session_directory / self.CLASSES_FILENAME

        if not classes_path.is_file():
            raise ValueError(
                f"Missing {self.CLASSES_FILENAME} in "
                f"'{session_directory.name}'."
            )

        classes: list[ClassDefinition] = []

        for line_number, raw_line in enumerate(
            classes_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue

            if "\t" not in raw_line:
                raise ValueError(
                    f"Invalid class entry on line {line_number} of "
                    f"'{classes_path}'."
                )

            class_id_text, class_name = raw_line.split("\t", 1)

            try:
                class_id = int(class_id_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid class ID on line {line_number} of "
                    f"'{classes_path}'."
                ) from error

            classes.append(
                ClassDefinition(class_id=class_id, name=class_name)
            )

        self._validate_classes(classes)
        return sorted(classes, key=lambda item: item.class_id)

    @staticmethod
    def _parse_model_paths(info: dict[str, str]) -> list[Path]:
        raw_model_sources = info.get("Model Sources", "").strip()

        if raw_model_sources:
            try:
                decoded = json.loads(raw_model_sources)
            except json.JSONDecodeError:
                decoded = []

            if isinstance(decoded, list):
                return [
                    Path(value)
                    for value in decoded
                    if isinstance(value, str) and value.strip()
                ]

        legacy_model_source = info.get("Model Source", "").strip()

        if legacy_model_source:
            return [Path(legacy_model_source)]

        return []

    def _validate_source_image_path(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> None:
        image_path = Path(image_path).resolve()
        source_directory = definition.image_directory.resolve()

        if image_path.parent != source_directory:
            raise ValueError(
                "Source images must be top-level files in the session "
                "image directory. Subdirectories are not used."
            )

    def _validate_source_workspace_separation(
        self,
        image_directory: Path,
    ) -> None:
        """Prevent session writes from occurring inside a source tree."""
        image_directory = Path(image_directory).resolve()
        workspace_root = self.workspace_root.resolve()
        open_sessions = self.open_sessions_directory.resolve()

        if (
            image_directory == workspace_root
            or image_directory in workspace_root.parents
        ):
            raise ValueError(
                "The save workspace cannot be located inside the source "
                "image directory. Choose a different source directory "
                "or save location so the source remains read-only."
            )

        if (
            image_directory == open_sessions
            or open_sessions in image_directory.parents
        ):
            raise ValueError(
                "A folder inside Open Sessions cannot be used as a "
                "source image directory."
            )

    def _validate_flat_image_names(
        self,
        image_directory: Path,
    ) -> None:
        image_paths = [
            path
            for path in image_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.casefold()
                in ImageService.SUPPORTED_EXTENSIONS
            )
        ]

        seen_stems: dict[str, str] = {}

        for image_path in image_paths:
            normalized_stem = image_path.stem.casefold()

            if normalized_stem in seen_stems:
                raise ValueError(
                    "The source directory contains image filenames "
                    "that would share the same YOLO annotation name: "
                    f"'{seen_stems[normalized_stem]}' and "
                    f"'{image_path.name}'. Rename one before creating "
                    "the session."
                )

            seen_stems[normalized_stem] = image_path.name

    def _session_name_exists(self, name: str) -> bool:
        normalized_name = name.casefold()

        return any(
            path.is_dir() and path.name.casefold() == normalized_name
            for path in self.open_sessions_directory.iterdir()
        )

    @staticmethod
    def _validate_session_name(name: str) -> str:
        cleaned_name = name.strip()

        if not cleaned_name:
            raise ValueError("Session name cannot be empty.")

        if cleaned_name in {".", ".."}:
            raise ValueError("Session name is invalid.")

        if any(character in cleaned_name for character in '<>:"/\\|?*'):
            raise ValueError(
                "Session name contains a character that cannot be "
                "used in a folder name."
            )

        return cleaned_name

    @staticmethod
    def _validate_model_paths(model_paths: list[Path]) -> None:
        for model_path in model_paths:
            if not model_path.is_file():
                raise ValueError(
                    f"Model file does not exist: {model_path}"
                )

    @staticmethod
    def _validate_classes(classes: list[ClassDefinition]) -> None:
        if not classes:
            raise ValueError("At least one annotation class is required.")

        seen_ids: set[int] = set()
        seen_names: set[str] = set()

        for class_definition in classes:
            if not isinstance(class_definition, ClassDefinition):
                raise TypeError(
                    "Classes must contain only ClassDefinition objects."
                )

            if class_definition.class_id in seen_ids:
                raise ValueError(
                    f"Duplicate class ID: {class_definition.class_id}"
                )

            normalized_name = class_definition.name.casefold()

            if normalized_name in seen_names:
                raise ValueError(
                    f"Duplicate class name: {class_definition.name}"
                )

            seen_ids.add(class_definition.class_id)
            seen_names.add(normalized_name)

    def _count_pooled_images(self, session_directory: Path) -> int:
        annotation_directory = self.annotations_directory(
            session_directory
        )
        image_directory = self.annotated_images_directory(
            session_directory
        )

        if not annotation_directory.is_dir() or not image_directory.is_dir():
            return 0

        copied_stems = {
            path.stem.casefold()
            for path in image_directory.iterdir()
            if path.is_file()
        }

        return sum(
            1
            for annotation_path in annotation_directory.glob("*.txt")
            if annotation_path.stem.casefold() in copied_stems
        )

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
        annotations: list,
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
        self._atomic_write_text(
            metadata_path,
            json.dumps(payload, indent=2) + "\n",
        )

    @staticmethod
    def _apply_annotation_metadata(
        metadata_path: Path,
        annotations: list,
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
        self._atomic_write_text(
            metadata_path,
            json.dumps(payload, indent=2) + "\n",
        )

    def _remove_copied_image_by_stem(
        self,
        definition: SessionDefinition,
        stem: str,
    ) -> None:
        image_directory = self.annotated_images_directory(
            definition.session_directory
        )
        normalized_stem = stem.casefold()

        for candidate in image_directory.iterdir():
            if (
                candidate.is_file()
                and candidate.stem.casefold() == normalized_stem
            ):
                candidate.unlink()

    def _assert_session_directory(self, path: Path) -> None:
        path = Path(path).resolve()
        open_sessions = self.open_sessions_directory.resolve()

        if path == open_sessions or open_sessions not in path.parents:
            raise ValueError(
                "Refusing to modify a path outside Open Sessions."
            )

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")

        try:
            temporary_path.write_text(text, encoding="utf-8")
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
