from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class SaveLocationDialog(QDialog):
    """First-run dialog for selecting the program save folder."""

    def __init__(
        self,
        default_location: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._default_location = Path(default_location)
        self._selected_location: Path | None = None

        self.setWindowTitle("Choose Save Location")
        self.setModal(True)
        self.setMinimumWidth(480)

        heading = QLabel("Choose Save Location")
        heading.setProperty("heading", "true")

        message = QLabel(
            "Choose where Model-Assisted Labeler should store its "
            "Open Sessions folder. You will only be asked again if "
            "this location no longer exists."
        )
        message.setWordWrap(True)
        message.setProperty("muted", "true")

        default_label = QLabel(
            f"Default: {self._default_location}"
        )
        default_label.setWordWrap(True)
        default_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        use_default_button = QPushButton("Use Default Location")
        use_default_button.setProperty("cta", "primary")
        use_default_button.setMinimumHeight(36)
        choose_button = QPushButton("Choose Another Location...")
        choose_button.setMinimumHeight(36)
        cancel_button = QPushButton("Exit")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(heading)
        layout.addWidget(message)
        layout.addWidget(default_label)
        layout.addSpacing(8)
        layout.addWidget(use_default_button)
        layout.addWidget(choose_button)
        layout.addSpacing(8)
        layout.addWidget(cancel_button)

        use_default_button.clicked.connect(self._use_default)
        choose_button.clicked.connect(self._choose_location)
        cancel_button.clicked.connect(self.reject)

    @property
    def selected_location(self) -> Path | None:
        return self._selected_location

    def _use_default(self) -> None:
        self._selected_location = self._default_location
        self.accept()

    def _choose_location(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            str(self._default_location.parent),
        )

        if not directory:
            return

        self._selected_location = Path(directory)
        self.accept()


class StartupDialog(QDialog):
    """Small launcher shown before the main application window."""

    CREATE_ACTION = "create"
    LOAD_ACTION = "load"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selected_action: str | None = None

        self.setWindowTitle("Model-Assisted Labeler")
        self.setModal(True)
        self.setFixedWidth(360)

        title = QLabel("Model-Assisted Labeler")
        title.setProperty("heading", "true")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Annotation Sessions")
        subtitle.setProperty("muted", "true")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        create_button = QPushButton("Create New Session")
        load_button = QPushButton("Load Session")
        exit_button = QPushButton("Exit")

        create_button.setProperty("cta", "primary")
        create_button.setMinimumHeight(42)
        load_button.setMinimumHeight(42)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addWidget(create_button)
        layout.addWidget(load_button)
        layout.addSpacing(4)
        layout.addWidget(exit_button)

        create_button.clicked.connect(
            lambda: self._select(self.CREATE_ACTION)
        )
        load_button.clicked.connect(
            lambda: self._select(self.LOAD_ACTION)
        )
        exit_button.clicked.connect(self.reject)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    def _select(self, action: str) -> None:
        self._selected_action = action
        self.accept()
