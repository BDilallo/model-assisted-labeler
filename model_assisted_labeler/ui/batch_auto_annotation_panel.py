from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class BatchAutoAnnotationPanel(QWidget):
    """
    Batch auto-annotation settings, shown as a top-bar dropdown panel.

    Lives inside a QMenu attached to the top bar's "Batch Auto
    Annotate" button. Settings persist between runs instead of being
    re-entered in a modal dialog each time.
    """

    MIN_CONFIDENCE_PERCENT = 0
    MAX_CONFIDENCE_PERCENT = 99
    DEFAULT_CONFIDENCE_PERCENT = 80

    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 64
    DEFAULT_BATCH_SIZE = 8

    DEVICE_AUTO = "Auto"
    DEVICE_CPU = "CPU"
    DEVICE_GPU = "GPU"

    run_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: transparent;")

        self._candidate_count = 0

        self._confidence_spin_box = QSpinBox()
        self._confidence_spin_box.setRange(
            self.MIN_CONFIDENCE_PERCENT,
            self.MAX_CONFIDENCE_PERCENT,
        )
        self._confidence_spin_box.setValue(
            self.DEFAULT_CONFIDENCE_PERCENT
        )
        self._confidence_spin_box.setSuffix("%")
        self._confidence_spin_box.setAlignment(
            Qt.AlignmentFlag.AlignRight
        )

        self._batch_size_spin_box = QSpinBox()
        self._batch_size_spin_box.setRange(
            self.MIN_BATCH_SIZE,
            self.MAX_BATCH_SIZE,
        )
        self._batch_size_spin_box.setValue(self.DEFAULT_BATCH_SIZE)
        self._batch_size_spin_box.setAlignment(
            Qt.AlignmentFlag.AlignRight
        )

        self._device_combo_box = QComboBox()
        self._device_combo_box.addItems(
            [self.DEVICE_AUTO, self.DEVICE_CPU, self.DEVICE_GPU]
        )

        self._candidate_label = QLabel()
        self._candidate_label.setProperty("muted", "true")
        self._candidate_label.setWordWrap(True)

        self._run_button = QPushButton("Run Batch Auto Annotate")
        self._run_button.setProperty("cta", "primary")
        self._run_button.setMinimumHeight(30)

        self._build_layout()
        self._run_button.clicked.connect(self.run_requested.emit)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        heading = QLabel("Batch Auto Annotate")
        heading.setProperty("heading", "true")
        layout.addWidget(heading)

        layout.addWidget(self._candidate_label)

        form = QFormLayout()
        form.setSpacing(6)

        confidence_row = QHBoxLayout()
        confidence_row.addWidget(self._confidence_spin_box, stretch=1)
        confidence_row.addWidget(
            self._help_button(
                "Only model boxes at or above this confidence are "
                "saved. Images with no qualifying boxes are left "
                "unsaved."
            )
        )
        form.addRow("Minimum confidence:", confidence_row)

        batch_size_row = QHBoxLayout()
        batch_size_row.addWidget(self._batch_size_spin_box, stretch=1)
        batch_size_row.addWidget(
            self._help_button(
                "Number of images sent to the model per inference "
                "call. Larger batches run faster on a GPU but use "
                "more memory. Lower this if a run fails or the "
                "device runs out of memory."
            )
        )
        form.addRow("Batch size:", batch_size_row)

        device_row = QHBoxLayout()
        device_row.addWidget(self._device_combo_box, stretch=1)
        device_row.addWidget(
            self._help_button(
                "Auto lets the model pick the best available device. "
                "Choose CPU or GPU to force one. GPU is unavailable "
                "if no CUDA-capable device was detected."
            )
        )
        form.addRow("Device:", device_row)

        layout.addLayout(form)
        layout.addWidget(self._run_button)

    @staticmethod
    def _help_button(tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText("?")
        button.setFixedSize(22, 22)
        button.setToolTip(tooltip)
        return button

    def set_candidate_count(self, count: int) -> None:
        self._candidate_count = max(0, count)
        image_word = "image" if self._candidate_count == 1 else "images"
        self._candidate_label.setText(
            f"{self._candidate_count} clean, unsaved {image_word} "
            "eligible for batch annotation."
        )

    def set_gpu_available(self, available: bool) -> None:
        gpu_index = self._device_combo_box.findText(self.DEVICE_GPU)

        if gpu_index == -1:
            return

        item_model = self._device_combo_box.model()
        item = item_model.item(gpu_index)

        if item is not None:
            item.setEnabled(available)

        if not available and self._device_combo_box.currentIndex() == (
            gpu_index
        ):
            self._device_combo_box.setCurrentText(self.DEVICE_AUTO)

        self._device_combo_box.setItemData(
            gpu_index,
            (
                "GPU"
                if available
                else "No CUDA-capable GPU was detected."
            ),
            Qt.ItemDataRole.ToolTipRole,
        )

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_button.setEnabled(enabled and self._candidate_count > 0)

    @property
    def confidence_percent(self) -> int:
        return self._confidence_spin_box.value()

    @property
    def confidence_threshold(self) -> float:
        return self.confidence_percent / 100.0

    @property
    def batch_size(self) -> int:
        return self._batch_size_spin_box.value()

    @property
    def device(self) -> str | int | None:
        selection = self._device_combo_box.currentText()

        if selection == self.DEVICE_CPU:
            return "cpu"

        if selection == self.DEVICE_GPU:
            return 0

        return None
