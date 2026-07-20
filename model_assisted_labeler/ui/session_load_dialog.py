from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from model_assisted_labeler.models.session_definition import SessionDefinition
from model_assisted_labeler.services.session_repository import SessionRepository


class SessionLoadDialog(QDialog):
    """Display saved sessions as a vertical list of selectable tiles."""

    def __init__(
        self,
        session_repository: SessionRepository,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_repository = session_repository
        self._selected_session: SessionDefinition | None = None

        self.setWindowTitle("Load Session")
        self.setModal(True)
        self.resize(680, 560)
        self.setMinimumSize(560, 420)

        self._tile_container = QWidget()
        self._tile_layout = QVBoxLayout(self._tile_container)
        self._tile_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._tile_container)

        cancel_button = QPushButton("Back")
        cancel_button.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Open Sessions"))
        layout.addWidget(scroll_area, stretch=1)
        layout.addWidget(cancel_button)

        self._refresh_tiles()

    @property
    def selected_session(self) -> SessionDefinition | None:
        return self._selected_session

    def _refresh_tiles(self) -> None:
        while self._tile_layout.count():
            item = self._tile_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        sessions = self._session_repository.list_sessions()

        if not sessions:
            empty_label = QLabel("No saved sessions were found.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._tile_layout.addWidget(empty_label)
            return

        for definition in sessions:
            self._tile_layout.addWidget(
                self._create_session_tile(definition)
            )

        self._tile_layout.addStretch(1)

    def _create_session_tile(
        self,
        definition: SessionDefinition,
    ) -> QFrame:
        tile = QFrame()
        tile.setFrameShape(QFrame.Shape.StyledPanel)

        name_label = QLabel(definition.name)
        name_label.setObjectName("sessionTileName")

        image_label = QLabel(
            f"Images: {definition.image_directory}"
        )
        image_label.setWordWrap(True)

        count_label = QLabel(
            "Annotated images: "
            f"{definition.total_images_annotated}"
        )

        details_layout = QVBoxLayout()
        details_layout.addWidget(name_label)
        details_layout.addWidget(image_label)
        details_layout.addWidget(count_label)

        open_button = QPushButton("Open")
        delete_button = QPushButton("Delete Session")

        open_button.clicked.connect(
            lambda checked=False, item=definition: self._open(item)
        )
        delete_button.clicked.connect(
            lambda checked=False, item=definition: self._delete(item)
        )

        button_layout = QVBoxLayout()
        button_layout.addWidget(open_button)
        button_layout.addWidget(delete_button)
        button_layout.addStretch(1)

        layout = QHBoxLayout(tile)
        layout.addLayout(details_layout, stretch=1)
        layout.addLayout(button_layout)
        return tile

    def _open(self, definition: SessionDefinition) -> None:
        self._selected_session = definition
        self.accept()

    def _delete(self, definition: SessionDefinition) -> None:
        typed_name, accepted = QInputDialog.getText(
            self,
            "Confirm Session Deletion",
            (
                "This permanently deletes the saved session folder. "
                "The source image directory will not be modified.\n\n"
                f"Type '{definition.name}' to confirm:"
            ),
        )

        if not accepted:
            return

        if typed_name != definition.name:
            QMessageBox.warning(
                self,
                "Name Does Not Match",
                "The session name did not match. Nothing was deleted.",
            )
            return

        try:
            self._session_repository.delete_session(definition)
        except Exception as error:
            QMessageBox.critical(self, "Deletion Error", str(error))
            return

        QMessageBox.information(
            self,
            "Session Deleted",
            f"Session '{definition.name}' was deleted.",
        )
        self._refresh_tiles()
