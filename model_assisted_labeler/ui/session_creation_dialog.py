from dataclasses import dataclass
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
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.models.annotation_session import ClassDefinition


@dataclass
class SessionCreationData:
    name: str
    image_directory: Path
    model_paths: list[Path]
    classes: list[ClassDefinition]


class SessionCreationDialog(QDialog):
    """Collect all information required to create a saved session."""

    SOURCE_ROLE = Qt.ItemDataRole.UserRole
    NAME_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(
        self,
        controller: AnnotationController,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._result_data: SessionCreationData | None = None

        self._model_path: Path | None = None
        self._model_classes: dict[str, ClassDefinition] = {}
        self._included_model_names: set[str] = set()
        self._manual_class_names: list[str] = []

        self._name_edit = QLineEdit()
        self._image_path_edit = QLineEdit()
        self._model_path_edit = QLineEdit()
        self._class_list = QListWidget()
        self._new_class_edit = QLineEdit()

        self._image_path_edit.setReadOnly(True)
        self._model_path_edit.setReadOnly(True)
        self._new_class_edit.setPlaceholderText("New class name")

        self.setWindowTitle("Create New Session")
        self.setModal(True)
        self.resize(760, 620)
        self.setMinimumSize(650, 520)

        self._build_layout()

    @property
    def result_data(self) -> SessionCreationData | None:
        return self._result_data

    def _build_layout(self) -> None:
        main_layout = QVBoxLayout(self)

        heading = QLabel("Create New Annotation Session")
        heading.setObjectName("sessionCreationHeading")
        main_layout.addWidget(heading)

        form_layout = QFormLayout()
        form_layout.addRow("Session name:", self._name_edit)

        image_row = QHBoxLayout()
        image_row.addWidget(self._image_path_edit, stretch=1)
        image_browse = QPushButton("Browse...")
        image_row.addWidget(image_browse)
        form_layout.addRow("Image directory:", image_row)

        image_note = QLabel(
            "Only image files directly inside this folder are used. "
            "Subfolders are never searched or modified."
        )
        image_note.setWordWrap(True)
        form_layout.addRow("", image_note)

        model_row = QHBoxLayout()
        model_row.addWidget(self._model_path_edit, stretch=1)
        model_browse = QPushButton("Browse...")
        model_clear = QPushButton("None")
        model_row.addWidget(model_browse)
        model_row.addWidget(model_clear)
        form_layout.addRow("Assistance model:", model_row)

        model_note = QLabel(
            "A model is optional. Model classes are imported with their "
            "original IDs. Unwanted model classes may be removed from "
            "the session class list."
        )
        model_note.setWordWrap(True)
        form_layout.addRow("", model_note)

        main_layout.addLayout(form_layout)

        class_label = QLabel("Classes used by this session")
        main_layout.addWidget(class_label)
        main_layout.addWidget(self._class_list, stretch=1)

        class_controls = QHBoxLayout()
        class_controls.addWidget(self._new_class_edit, stretch=1)
        add_class_button = QPushButton("Add Class")
        remove_class_button = QPushButton("Remove Selected")
        class_controls.addWidget(add_class_button)
        class_controls.addWidget(remove_class_button)
        main_layout.addLayout(class_controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        create_button = buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )

        if create_button is not None:
            create_button.setText("Create Session")

        main_layout.addWidget(buttons)

        image_browse.clicked.connect(self._browse_image_directory)
        model_browse.clicked.connect(self._browse_model)
        model_clear.clicked.connect(self._clear_model)
        add_class_button.clicked.connect(self._add_manual_class)
        remove_class_button.clicked.connect(self._remove_selected_class)
        self._new_class_edit.returnPressed.connect(
            self._add_manual_class
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

    def _browse_image_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Source Image Directory",
        )

        if directory:
            self._image_path_edit.setText(directory)

    def _browse_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Detection Model",
            "",
            "Detection Models (*.pt *.onnx *.engine);;All Files (*)",
        )

        if not filename:
            return

        model_path = Path(filename)

        try:
            model_classes = self._controller.inspect_model_classes(
                model_path
            )
        except Exception as error:
            QMessageBox.critical(self, "Model Error", str(error))
            return

        self._model_path = model_path
        self._model_path_edit.setText(str(model_path))
        self._model_classes = {
            item.name.casefold(): item for item in model_classes
        }
        self._included_model_names = set(self._model_classes)

        overlapping_manual_names = [
            name
            for name in self._manual_class_names
            if name.casefold() in self._model_classes
        ]

        if overlapping_manual_names:
            overlap_set = {
                name.casefold() for name in overlapping_manual_names
            }
            self._manual_class_names = [
                name
                for name in self._manual_class_names
                if name.casefold() not in overlap_set
            ]

            QMessageBox.information(
                self,
                "Model Classes Used",
                (
                    "The following class names also exist in the model. "
                    "The model versions and IDs will be used:\n\n"
                    + "\n".join(overlapping_manual_names)
                ),
            )

        self._refresh_class_list()

    def _clear_model(self) -> None:
        self._model_path = None
        self._model_path_edit.clear()
        self._model_classes.clear()
        self._included_model_names.clear()
        self._controller.clear_model()
        self._refresh_class_list()

    def _add_manual_class(self) -> None:
        class_name = self._new_class_edit.text().strip()

        if not class_name:
            return

        normalized_name = class_name.casefold()

        if normalized_name in self._model_classes:
            self._included_model_names.add(normalized_name)
            model_class = self._model_classes[normalized_name]
            QMessageBox.information(
                self,
                "Model Class Used",
                (
                    f"'{class_name}' is already defined by the selected "
                    "model. The model version (ID "
                    f"{model_class.class_id}) will be used."
                ),
            )
            self._new_class_edit.clear()
            self._refresh_class_list()
            return

        if any(
            existing.casefold() == normalized_name
            for existing in self._manual_class_names
        ):
            QMessageBox.warning(
                self,
                "Duplicate Class",
                f"Class '{class_name}' already exists.",
            )
            return

        self._manual_class_names.append(class_name)
        self._new_class_edit.clear()
        self._refresh_class_list()

    def _remove_selected_class(self) -> None:
        item = self._class_list.currentItem()

        if item is None:
            return

        source = item.data(self.SOURCE_ROLE)
        class_name = str(item.data(self.NAME_ROLE))
        normalized_name = class_name.casefold()

        if source == "model":
            self._included_model_names.discard(normalized_name)
        else:
            self._manual_class_names = [
                name
                for name in self._manual_class_names
                if name.casefold() != normalized_name
            ]

        self._refresh_class_list()

    def _refresh_class_list(self) -> None:
        self._class_list.clear()

        included_model_classes = [
            class_definition
            for normalized_name, class_definition
            in self._model_classes.items()
            if normalized_name in self._included_model_names
        ]

        for class_definition in sorted(
            included_model_classes,
            key=lambda item: item.class_id,
        ):
            item = QListWidgetItem(
                f"{class_definition.class_id}: "
                f"{class_definition.name}  [model]"
            )
            item.setData(self.SOURCE_ROLE, "model")
            item.setData(self.NAME_ROLE, class_definition.name)
            self._class_list.addItem(item)

        for class_name in self._manual_class_names:
            item = QListWidgetItem(f"Auto: {class_name}  [manual]")
            item.setData(self.SOURCE_ROLE, "manual")
            item.setData(self.NAME_ROLE, class_name)
            self._class_list.addItem(item)

    def _build_class_definitions(self) -> list[ClassDefinition]:
        model_classes = [
            class_definition
            for normalized_name, class_definition
            in self._model_classes.items()
            if normalized_name in self._included_model_names
        ]
        reserved_model_ids = {
            class_definition.class_id
            for class_definition in self._model_classes.values()
        }
        used_ids = {
            class_definition.class_id
            for class_definition in model_classes
        }

        manual_classes: list[ClassDefinition] = []
        next_candidate = 0

        for class_name in self._manual_class_names:
            while (
                next_candidate in reserved_model_ids
                or next_candidate in used_ids
            ):
                next_candidate += 1

            manual_classes.append(
                ClassDefinition(next_candidate, class_name)
            )
            used_ids.add(next_candidate)
            next_candidate += 1

        return sorted(
            model_classes + manual_classes,
            key=lambda item: item.class_id,
        )

    def _validate_and_accept(self) -> None:
        session_name = self._name_edit.text().strip()
        image_path_text = self._image_path_edit.text().strip()

        if not session_name:
            QMessageBox.warning(
                self,
                "Missing Session Name",
                "Enter a name for the session.",
            )
            return

        if not image_path_text:
            QMessageBox.warning(
                self,
                "Missing Image Directory",
                "Select the image directory to annotate.",
            )
            return

        image_directory = Path(image_path_text)

        if not image_directory.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Image Directory",
                "The selected image directory does not exist.",
            )
            return

        classes = self._build_class_definitions()

        if not classes:
            QMessageBox.warning(
                self,
                "Missing Classes",
                "Add at least one class before creating the session.",
            )
            return

        model_paths = [self._model_path] if self._model_path else []
        self._result_data = SessionCreationData(
            name=session_name,
            image_directory=image_directory,
            model_paths=model_paths,
            classes=classes,
        )
        self.accept()
