from pathlib import Path

from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.models.class_definition import ClassDefinition
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.repositories import session_paths
from model_assisted_labeler.repositories.annotation_pool_repository import (
    AnnotationPoolRepository,
)
from model_assisted_labeler.repositories.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.repositories.session_document_repository import (
    SessionAlreadyExistsError,
    SessionDocumentRepository,
)

__all__ = ["SessionAlreadyExistsError", "SessionRepository"]


class SessionRepository:
    """
    Owns all persistent session reads and writes.

    Source image directories are treated as read-only. Every mutation
    performed by this class is constrained to the program-owned
    ``Open Sessions`` directory. A thin facade over
    ``SessionDocumentRepository`` (session-level documents) and
    ``AnnotationPoolRepository`` (pooled annotations).
    """

    OPEN_SESSIONS_DIRECTORY = SessionDocumentRepository.OPEN_SESSIONS_DIRECTORY
    SESSION_INFO_FILENAME = SessionDocumentRepository.SESSION_INFO_FILENAME
    CLASSES_FILENAME = SessionDocumentRepository.CLASSES_FILENAME
    ANNOTATED_IMAGES_DIRECTORY = session_paths.ANNOTATED_IMAGES_DIRECTORY
    ANNOTATIONS_DIRECTORY = session_paths.ANNOTATIONS_DIRECTORY
    ANNOTATION_METADATA_DIRECTORY = (
        session_paths.ANNOTATION_METADATA_DIRECTORY
    )

    def __init__(
        self,
        workspace_root: Path,
        annotation_store: YoloAnnotationStore,
    ) -> None:
        self.annotation_store = annotation_store

        self._document_repository = SessionDocumentRepository(
            workspace_root
        )
        self._pool_repository = AnnotationPoolRepository(
            annotation_store=annotation_store,
            document_repository=self._document_repository,
        )

    @property
    def workspace_root(self) -> Path:
        return self._document_repository.workspace_root

    @property
    def open_sessions_directory(self) -> Path:
        return self._document_repository.open_sessions_directory

    # -- session documents ------------------------------------------------

    def create_session(
        self,
        name: str,
        image_directory: Path,
        model_paths: list[Path],
        classes: list[ClassDefinition],
    ) -> SessionDefinition:
        return self._document_repository.create_session(
            name=name,
            image_directory=image_directory,
            model_paths=model_paths,
            classes=classes,
        )

    def list_sessions(self) -> list[SessionDefinition]:
        return self._document_repository.list_sessions()

    def load_session(
        self,
        session: str | Path,
    ) -> SessionDefinition:
        return self._document_repository.load_session(session)

    def delete_session(self, definition: SessionDefinition) -> None:
        self._document_repository.delete_session(definition)

    def update_paths(
        self,
        definition: SessionDefinition,
        image_directory: Path,
        model_paths: list[Path],
    ) -> None:
        self._document_repository.update_paths(
            definition,
            image_directory=image_directory,
            model_paths=model_paths,
        )

    def update_last_image(
        self,
        definition: SessionDefinition,
        filename: str | None,
    ) -> None:
        self._document_repository.update_last_image(definition, filename)

    def save_session_info(
        self,
        definition: SessionDefinition,
    ) -> None:
        self._document_repository.save_session_info(definition)

    def save_classes(self, definition: SessionDefinition) -> None:
        self._document_repository.save_classes(definition)

    # -- session paths ----------------------------------------------------

    def annotations_directory(self, session_directory: Path) -> Path:
        return session_paths.annotations_directory(session_directory)

    def annotated_images_directory(
        self,
        session_directory: Path,
    ) -> Path:
        return session_paths.annotated_images_directory(session_directory)

    def annotation_metadata_directory(
        self,
        session_directory: Path,
    ) -> Path:
        return session_paths.annotation_metadata_directory(
            session_directory
        )

    def annotation_path_for(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> Path:
        return session_paths.annotation_path_for(definition, image_path)

    def annotated_image_path_for(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> Path:
        return session_paths.annotated_image_path_for(
            definition,
            image_path,
        )

    def annotation_metadata_path_for(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> Path:
        return session_paths.annotation_metadata_path_for(
            definition,
            image_path,
        )

    # -- annotation pool ----------------------------------------------------

    def load_annotations(
        self,
        definition: SessionDefinition,
        image_record: ImageRecord,
    ) -> list[BoundingBox]:
        return self._pool_repository.load_annotations(
            definition,
            image_record,
        )

    def image_is_in_pool(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> bool:
        return self._pool_repository.image_is_in_pool(
            definition,
            image_path,
        )

    def save_image_to_pool(
        self,
        definition: SessionDefinition,
        image_record: ImageRecord,
        refresh_session_info: bool = True,
    ) -> None:
        self._pool_repository.save_image_to_pool(
            definition,
            image_record,
            refresh_session_info=refresh_session_info,
        )

    def remove_image_from_pool(
        self,
        definition: SessionDefinition,
        image_path: Path,
    ) -> None:
        self._pool_repository.remove_image_from_pool(
            definition,
            image_path,
        )

    def clear_annotation_pool(
        self,
        definition: SessionDefinition,
    ) -> None:
        self._pool_repository.clear_pool(definition)

    def class_usage_filenames(
        self,
        definition: SessionDefinition,
        class_id: int,
    ) -> list[str]:
        return self._pool_repository.class_usage_filenames(
            definition,
            class_id,
        )

    def remove_class_from_pool_annotations(
        self,
        definition: SessionDefinition,
        class_id: int,
        delete_referenced_images: bool,
    ) -> set[str]:
        return self._pool_repository.remove_class_from_pool_annotations(
            definition=definition,
            class_id=class_id,
            delete_referenced_images=delete_referenced_images,
        )
