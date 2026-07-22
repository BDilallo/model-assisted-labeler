from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)


class BatchAutoAnnotationDialog(QDialog):
    """Confirm batch annotation and collect its confidence threshold."""

    DEFAULT_CONFIDENCE_PERCENT = 80
    MINIMUM_CONFIDENCE_PERCENT = 0
    MAXIMUM_CONFIDENCE_PERCENT = 99

    def __init__(
        self,
        image_count: int,
        parent=None,
    ) -> None:
        super().__init__(parent)

        if image_count < 0:
            raise ValueError("Image count cannot be negative.")

        self._confidence_spin_box = QSpinBox()
        self._confidence_spin_box.setRange(
            self.MINIMUM_CONFIDENCE_PERCENT,
            self.MAXIMUM_CONFIDENCE_PERCENT,
        )
        self._confidence_spin_box.setValue(
            self.DEFAULT_CONFIDENCE_PERCENT
        )
        self._confidence_spin_box.setSuffix("%")
        self._confidence_spin_box.setAlignment(
            Qt.AlignmentFlag.AlignRight
        )

        self.setWindowTitle("Batch Auto Annotate")
        self.setModal(True)
        self.setFixedWidth(430)

        image_word = "image" if image_count == 1 else "images"
        prompt = QLabel(
            "Automatically Annotate and Save "
            f"{image_count} {image_word}?"
        )
        prompt.setWordWrap(True)

        confidence_label = QLabel("Minimum confidence:")
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setFixedSize(22, 22)
        help_button.setToolTip(
            "Only model boxes at or above this confidence are saved. "
            "If an image has no qualifying boxes, that image is left "
            "unsaved. The value may range from 0% through 99%."
        )

        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(
            self._confidence_spin_box,
            stretch=1,
        )
        confidence_layout.addWidget(help_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        confirm_button = buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )

        if confirm_button is not None:
            confirm_button.setText("Confirm")
            confirm_button.setDefault(True)

        layout = QVBoxLayout(self)
        layout.addWidget(prompt)
        layout.addSpacing(8)
        layout.addLayout(confidence_layout)
        layout.addSpacing(8)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    @property
    def confidence_percent(self) -> int:
        """Return the selected whole-number confidence percentage."""
        return self._confidence_spin_box.value()

    @property
    def confidence_threshold(self) -> float:
        """Return the selected confidence as a zero-to-one value."""
        return self.confidence_percent / 100.0
