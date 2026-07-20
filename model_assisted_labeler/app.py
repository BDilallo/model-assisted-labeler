import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.services.application_settings import (
    ApplicationSettingsRepository,
)
from model_assisted_labeler.services.image_service import ImageService
from model_assisted_labeler.services.model_runner import (
    UltralyticsDetectionRunner,
)
from model_assisted_labeler.services.session_repository import (
    SessionAlreadyExistsError,
    SessionRepository,
)
from model_assisted_labeler.ui.main_window import MainWindow
from model_assisted_labeler.ui.session_creation_dialog import (
    SessionCreationDialog,
)
from model_assisted_labeler.ui.session_load_dialog import SessionLoadDialog
from model_assisted_labeler.ui.session_review_dialog import (
    SessionReviewDialog,
)
from model_assisted_labeler.ui.startup_dialog import (
    SaveLocationDialog,
    StartupDialog,
)


def _resolve_workspace_root(
    settings_repository: ApplicationSettingsRepository,
) -> Path | None:
    existing_location = settings_repository.get_workspace_root()

    if existing_location is not None:
        return existing_location

    location_dialog = SaveLocationDialog(
        settings_repository.default_workspace_root()
    )

    if location_dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    selected_location = location_dialog.selected_location

    if selected_location is None:
        return None

    try:
        selected_location.mkdir(parents=True, exist_ok=True)
        settings_repository.set_workspace_root(selected_location)
    except OSError as error:
        QMessageBox.critical(
            None,
            "Save Location Error",
            f"Could not create the save folder:\n{error}",
        )
        return None

    return selected_location


def _build_controller(
    session_repository: SessionRepository,
    annotation_store: YoloAnnotationStore,
) -> AnnotationController:
    image_service = ImageService()
    session_builder = AnnotationSessionBuilder(
        image_service=image_service,
        session_repository=session_repository,
    )
    model_runner = UltralyticsDetectionRunner()

    return AnnotationController(
        session_builder=session_builder,
        annotation_store=annotation_store,
        model_runner=model_runner,
        session_repository=session_repository,
    )


def _choose_session(
    controller: AnnotationController,
    session_repository: SessionRepository,
) -> SessionDefinition | None:
    while True:
        startup_dialog = StartupDialog()

        if startup_dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        if startup_dialog.selected_action == StartupDialog.CREATE_ACTION:
            creation_dialog = SessionCreationDialog(controller)

            if creation_dialog.exec() != QDialog.DialogCode.Accepted:
                continue

            creation_data = creation_dialog.result_data

            if creation_data is None:
                continue

            try:
                return session_repository.create_session(
                    name=creation_data.name,
                    image_directory=creation_data.image_directory,
                    model_paths=creation_data.model_paths,
                    classes=creation_data.classes,
                )
            except SessionAlreadyExistsError:
                QMessageBox.warning(
                    None,
                    "Session Exists",
                    "Session already exists",
                )
            except Exception as error:
                QMessageBox.critical(
                    None,
                    "Session Creation Error",
                    str(error),
                )

        elif startup_dialog.selected_action == StartupDialog.LOAD_ACTION:
            load_dialog = SessionLoadDialog(session_repository)

            if load_dialog.exec() != QDialog.DialogCode.Accepted:
                continue

            definition = load_dialog.selected_session

            if definition is None:
                continue

            review_dialog = SessionReviewDialog(
                definition=definition,
                session_repository=session_repository,
                controller=controller,
            )

            if review_dialog.exec() == QDialog.DialogCode.Accepted:
                return definition


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("Model-Assisted Labeler")
    application.setOrganizationName("Model-Assisted Labeler")

    settings_repository = ApplicationSettingsRepository()
    workspace_root = _resolve_workspace_root(settings_repository)

    if workspace_root is None:
        return 0

    annotation_store = YoloAnnotationStore()
    session_repository = SessionRepository(
        workspace_root=workspace_root,
        annotation_store=annotation_store,
    )
    controller = _build_controller(
        session_repository,
        annotation_store,
    )

    definition = _choose_session(controller, session_repository)

    if definition is None:
        return 0

    try:
        controller.open_session_definition(definition)
    except Exception as error:
        QMessageBox.critical(
            None,
            "Session Open Error",
            str(error),
        )
        return 1

    main_window = MainWindow(controller)
    main_window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
