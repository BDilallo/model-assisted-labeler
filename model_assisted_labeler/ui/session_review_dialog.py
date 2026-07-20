from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.session_repository import SessionRepository


class SessionReviewDialog(QDialog):
    """Review and repair saved paths before loading a session."""

    def __init__(
        self,
        definition: SessionDefinition,
        session_repository: SessionRepository,
        controller: AnnotationController,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._definition = definition
        self._session_repository = session_repository
        self._controller = controller

        self._image_path_edit = QLineEdit(
            str(definition.image_directory)
        )
        self._model_path_edit = QLineEdit(
            str(definition.primary_model_path or "")
        )
        self._image_path_edit.setReadOnly(True)
        self._model_path_edit.setReadOnly(True)

        self.setWindowTitle(f"Review Session - {definition.name}")
        self.setModal(True)
        self.resize(700, 520)

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        name_label = QLabel(self._definition.name)
        name_label.setObjectName("sessionReviewName")
        layout.addWidget(name_label)

        explanation = QLabel(
            "Review the saved locations before opening. Missing paths "
            "can be corrected here. Source image folders are always "
            "read-only and subfolders are not searched."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()

        image_row = QHBoxLayout()
        image_row.addWidget(self._image_path_edit, stretch=1)
        image_browse = QPushButton("Change...")
        image_row.addWidget(image_browse)
        form.addRow("Image directory:", image_row)

        model_row = QHBoxLayout()
        model_row.addWidget(self._model_path_edit, stretch=1)
        model_browse = QPushButton("Change...")
        model_clear = QPushButton("None")
        model_row.addWidget(model_browse)
        model_row.addWidget(model_clear)
        form.addRow("Assistance model:", model_row)

        layout.addLayout(form)

        class_list = QListWidget()

        for class_definition in sorted(
            self._definition.classes,
            key=lambda item: item.class_id,
        ):
            class_list.addItem(
                f"{class_definition.class_id}: {class_definition.name}"
            )

        layout.addWidget(QLabel("Session classes:"))
        layout.addWidget(class_list, stretch=1)

        count_label = QLabel(
            "Annotated images currently in pool: "
            f"{self._definition.total_images_annotated}"
        )
        count_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(count_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open
            | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        image_browse.clicked.connect(self._browse_image_directory)
        model_browse.clicked.connect(self._browse_model)
        model_clear.clicked.connect(self._model_path_edit.clear)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

    def _browse_image_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Source Image Directory",
            self._image_path_edit.text(),
        )

        if directory:
            self._image_path_edit.setText(directory)

    def _browse_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Detection Model",
            self._model_path_edit.text(),
            "Detection Models (*.pt *.onnx *.engine);;All Files (*)",
        )

        if filename:
            self._model_path_edit.setText(filename)

    def _validate_and_accept(self) -> None:
        image_directory = Path(self._image_path_edit.text().strip())
        model_text = self._model_path_edit.text().strip()
        model_paths = [Path(model_text)] if model_text else []

        if not image_directory.is_dir():
            QMessageBox.warning(
                self,
                "Missing Image Directory",
                "Select an existing source image directory.",
            )
            return

        if model_paths and not model_paths[0].is_file():
            QMessageBox.warning(
                self,
                "Missing Model",
                "Select an existing model file or choose None.",
            )
            return

        if model_paths:
            try:
                model_classes = self._controller.inspect_model_classes(
                    model_paths[0]
                )
            except Exception as error:
                QMessageBox.critical(self, "Model Error", str(error))
                return

            session_by_id = {
                item.class_id: item for item in self._definition.classes
            }
            session_by_name = {
                item.name.casefold(): item
                for item in self._definition.classes
            }
            conflicts: list[str] = []

            for model_class in model_classes:
                id_match = session_by_id.get(model_class.class_id)
                name_match = session_by_name.get(
                    model_class.name.casefold()
                )

                if (
                    id_match is not None
                    and id_match.name.casefold()
                    != model_class.name.casefold()
                ):
                    conflicts.append(
                        f"ID {model_class.class_id} is "
                        f"'{id_match.name}' in the session but "
                        f"'{model_class.name}' in the model."
                    )

                if (
                    name_match is not None
                    and name_match.class_id != model_class.class_id
                ):
                    conflicts.append(
                        f"'{model_class.name}' uses ID "
                        f"{name_match.class_id} in the session but "
                        f"{model_class.class_id} in the model."
                    )

            if conflicts:
                QMessageBox.critical(
                    self,
                    "Model Class Conflict",
                    (
                        "The replacement model is not compatible with "
                        "the saved session classes:\n\n"
                        + "\n".join(conflicts)
                    ),
                )
                return
        else:
            self._controller.clear_model()

        try:
            self._session_repository.update_paths(
                definition=self._definition,
                image_directory=image_directory,
                model_paths=model_paths,
            )
        except Exception as error:
            QMessageBox.critical(self, "Session Error", str(error))
            return

        self.accept()
