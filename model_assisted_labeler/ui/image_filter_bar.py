from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

from model_assisted_labeler.models.annotation_session import ClassDefinition


class ImageFilterBar(QWidget):
    """Select and describe the temporary image-review filter."""

    filter_changed = Signal()

    FILTER_OPTIONS = (
        ("all", "All Images"),
        ("unsaved", "Unsaved Images"),
        ("saved", "Saved Images"),
        ("no_boxes", "Images With No Boxes"),
        ("unsaved_changes", "Images With Unsaved Changes"),
        ("confidence_below", "Confidence Below X%"),
        ("confidence_above", "Confidence Above X%"),
        ("manual", "Manually Annotated Images"),
        ("model_only", "Model-Only Annotations"),
        ("edited_model", "Edited Model Annotations"),
        ("multiple_boxes", "Multiple Boxes"),
        ("single_box", "Single Box"),
        ("contains_class", "Contains Class"),
        ("missing_class", "Missing Class"),
    )

    CONFIDENCE_FILTERS = {
        "confidence_below",
        "confidence_above",
    }
    CLASS_FILTERS = {
        "contains_class",
        "missing_class",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._filter_combo = QComboBox()
        self._confidence_label = QLabel("Confidence:")
        self._confidence_spin = QSpinBox()
        self._confidence_help = QToolButton()
        self._class_label = QLabel("Class:")
        self._class_combo = QComboBox()
        self._result_label = QLabel("Showing 0 of 0")

        self._configure_widgets()
        self._build_layout()
        self._connect_signals()
        self._update_conditional_controls()

    @property
    def filter_key(self) -> str:
        value = self._filter_combo.currentData()
        return str(value) if value is not None else "all"

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_spin.value() / 100.0

    @property
    def class_id(self) -> int | None:
        value = self._class_combo.currentData()

        if value is None:
            return None

        return int(value)

    @property
    def is_all_images(self) -> bool:
        return self.filter_key == "all"

    def set_classes(
        self,
        classes: list[ClassDefinition],
    ) -> None:
        """Refresh class choices while preserving the selected ID."""
        previous_class_id = self.class_id
        self._class_combo.blockSignals(True)

        try:
            self._class_combo.clear()

            for class_definition in sorted(
                classes,
                key=lambda item: item.class_id,
            ):
                self._class_combo.addItem(
                    f"{class_definition.class_id}: "
                    f"{class_definition.name}",
                    class_definition.class_id,
                )

            if previous_class_id is not None:
                previous_index = self._class_combo.findData(
                    previous_class_id
                )

                if previous_index >= 0:
                    self._class_combo.setCurrentIndex(previous_index)

        finally:
            self._class_combo.blockSignals(False)

    def set_result_count(
        self,
        matching_images: int,
        total_images: int,
    ) -> None:
        self._result_label.setText(
            f"Showing {matching_images} of {total_images}"
        )

    def _configure_widgets(self) -> None:
        for filter_key, display_name in self.FILTER_OPTIONS:
            self._filter_combo.addItem(display_name, filter_key)

        self._filter_combo.setMinimumContentsLength(25)
        self._filter_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )

        self._confidence_spin.setRange(0, 99)
        self._confidence_spin.setValue(80)
        self._confidence_spin.setSuffix("%")
        self._confidence_spin.setToolTip(
            "The image confidence is the lowest stored confidence among "
            "its model-generated boxes. Images without stored model "
            "confidence are excluded from confidence filters."
        )

        self._confidence_help.setText("?")
        self._confidence_help.setAutoRaise(True)
        self._confidence_help.setToolTip(
            "Confidence filters review an image using its lowest stored "
            "model-box confidence. One low-confidence box is enough for "
            "the image to appear in Confidence Below."
        )
        self._confidence_help.setFixedSize(22, 22)

        self._class_combo.setMinimumContentsLength(16)
        self._result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

    def _build_layout(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Filter:"))
        layout.addWidget(self._filter_combo)
        layout.addSpacing(10)
        layout.addWidget(self._confidence_label)
        layout.addWidget(self._confidence_spin)
        layout.addWidget(self._confidence_help)
        layout.addSpacing(10)
        layout.addWidget(self._class_label)
        layout.addWidget(self._class_combo)
        layout.addStretch(1)
        layout.addWidget(self._result_label)

    def _connect_signals(self) -> None:
        self._filter_combo.currentIndexChanged.connect(
            lambda _index: self._handle_filter_selection_changed()
        )
        self._confidence_spin.valueChanged.connect(
            lambda _value: self.filter_changed.emit()
        )
        self._class_combo.currentIndexChanged.connect(
            lambda _index: self.filter_changed.emit()
        )

    def _handle_filter_selection_changed(self) -> None:
        self._update_conditional_controls()
        self.filter_changed.emit()

    def _update_conditional_controls(self) -> None:
        confidence_visible = self.filter_key in self.CONFIDENCE_FILTERS
        class_visible = self.filter_key in self.CLASS_FILTERS

        self._confidence_label.setVisible(confidence_visible)
        self._confidence_spin.setVisible(confidence_visible)
        self._confidence_help.setVisible(confidence_visible)
        self._class_label.setVisible(class_visible)
        self._class_combo.setVisible(class_visible)
