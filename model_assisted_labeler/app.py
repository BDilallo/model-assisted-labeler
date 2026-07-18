import sys

from PySide6.QtWidgets import QApplication

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.services.annotation_session_builder import (
    AnnotationSessionBuilder,
)
from model_assisted_labeler.services.annotation_store import (
    YoloAnnotationStore,
)
from model_assisted_labeler.services.image_service import ImageService
from model_assisted_labeler.services.model_runner import (
    UltralyticsDetectionRunner,
)
from model_assisted_labeler.ui.main_window import MainWindow


def build_controller() -> AnnotationController:
    """
    Construct the services used by the annotation application.

    Keeping object creation here makes the dependencies explicit and
    leaves MainWindow responsible only for interface behavior.
    """
    image_service = ImageService()
    annotation_store = YoloAnnotationStore()

    session_builder = AnnotationSessionBuilder(
        image_service=image_service,
        annotation_store=annotation_store,
    )

    model_runner = UltralyticsDetectionRunner(
        confidence_threshold=0.25,
        device=None,
    )

    return AnnotationController(
        session_builder=session_builder,
        annotation_store=annotation_store,
        model_runner=model_runner,
    )


def main() -> int:
    """
    Create and start the Qt application.

    Returns:
        The application's exit code.
    """
    application = QApplication(sys.argv)

    application.setApplicationName(
        "Model-Assisted Labeler"
    )
    application.setOrganizationName(
        "Model-Assisted Labeler"
    )

    controller = build_controller()
    main_window = MainWindow(controller)

    main_window.show()

    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
