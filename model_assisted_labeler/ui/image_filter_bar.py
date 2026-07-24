from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QToolButton,
)

from model_assisted_labeler.models.annotation_session import ClassDefinition


class ImageFilterBar(QFrame):
    """Top-bar controls for narrowing the session review list."""

    filter_changed = Signal()

    FILTER_OPTIONS = (
        ("all", "All Images"),
        ("unsaved", "Unsaved Images"),
        ("saved", "Saved Images"),
        ("no_boxes", "Images With No Boxes"),
        ("unsaved_changes", "Images With Unsaved Changes"),
        ("confidence_below", "Confidence Below"),
        ("confidence_at_or_above", "Confidence At or Above"),
        ("manual", "Manually Annotated Images"),
        ("model_only", "Model-Only Annotations"),
        ("edited_model", "Edited Model Annotations"),
        ("single_box", "Single Box"),
        ("multiple_boxes", "Multiple Boxes"),
        ("missing_class", "Missing Selected Class"),
    )

    CONFIDENCE_FILTERS = {
        "confidence_below",
        "confidence_at_or_above",
    }

    FILTERS_WITHOUT_CLASS_REFINEMENT = {
        "no_boxes",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._filter_combo = QComboBox()
        self._confidence_spin = QSpinBox()
        self._confidence_help = QToolButton()
        self._class_combo = QComboBox()
        self._result_label = QLabel("Showing 0 of 0")

        self._configure_frame()
        self._configure_widgets()
        self._build_layout()
        self._connect_signals()
        self._update_control_states()

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
        return self.filter_key == "all" and self.class_id is None

    def set_classes(
        self,
        classes: list[ClassDefinition],
    ) -> None:
        """Refresh the class dropdown while preserving its selection."""
        previous_class_id = self.class_id
        self._class_combo.blockSignals(True)

        try:
            self._class_combo.clear()
            self._class_combo.addItem("Any", None)

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

            self._ensure_required_class_selected()

        finally:
            self._class_combo.blockSignals(False)

        self._update_control_states()

    def set_result_count(
        self,
        matching_images: int,
        total_images: int,
    ) -> None:
        if matching_images < 0 or total_images < 0:
            raise ValueError("Image counts cannot be negative.")

        if matching_images > total_images:
            raise ValueError(
                "Matching image count cannot exceed total image count."
            )

        self._result_label.setText(
            f"Showing {matching_images} of {total_images}"
        )

    def reset(self) -> None:
        """Restore the default filter state without replacing classes."""
        widgets = (
            self._filter_combo,
            self._confidence_spin,
            self._class_combo,
        )

        for widget in widgets:
            widget.blockSignals(True)

        try:
            all_index = self._filter_combo.findData("all")

            if all_index >= 0:
                self._filter_combo.setCurrentIndex(all_index)

            self._confidence_spin.setValue(80)
            any_index = self._class_combo.findData(None)

            if any_index >= 0:
                self._class_combo.setCurrentIndex(any_index)

        finally:
            for widget in widgets:
                widget.blockSignals(False)

        self._update_control_states()
        self.filter_changed.emit()

    def _configure_frame(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("imageFilterBar")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    def _configure_widgets(self) -> None:
        for filter_key, display_name in self.FILTER_OPTIONS:
            self._filter_combo.addItem(display_name, filter_key)

        self._filter_combo.setMinimumContentsLength(24)
        self._filter_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._filter_combo.setToolTip(
            "Choose which images are included when using Back, Next, "
            "and Save & Next. Filtering never deletes or reorders images."
        )

        self._confidence_spin.setRange(0, 99)
        self._confidence_spin.setValue(80)
        self._confidence_spin.setSuffix("%")
        self._confidence_spin.setMinimumWidth(78)
        self._confidence_spin.setToolTip(
            "Used by the confidence filters. Image confidence is the "
            "lowest stored confidence among matching model-generated "
            "boxes. Images without stored model confidence are excluded."
        )

        self._confidence_help.setText("?")
        self._confidence_help.setAutoRaise(True)
        self._confidence_help.setFixedSize(22, 22)
        self._confidence_help.setToolTip(
            "Confidence is evaluated using the lowest-confidence model "
            "box in the image. When a class is selected, only model boxes "
            "of that class are considered. The value must remain below "
            "100%."
        )

        self._class_combo.addItem("Any", None)
        self._class_combo.setMinimumContentsLength(16)
        self._class_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._class_combo.setToolTip(
            "Optionally narrow the selected filter to images containing "
            "this class. Missing Selected Class reverses that check."
        )

        self._result_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter
        )
        self._result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._result_label.setMinimumWidth(130)

    def _build_layout(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Filter:"))
        layout.addWidget(self._filter_combo)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Confidence:"))
        layout.addWidget(self._confidence_spin)
        layout.addWidget(self._confidence_help)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Class:"))
        layout.addWidget(self._class_combo)
        layout.addStretch(1)
        layout.addWidget(self._result_label)

    def _connect_signals(self) -> None:
        self._filter_combo.currentIndexChanged.connect(
            self._handle_filter_selection_changed
        )
        self._confidence_spin.valueChanged.connect(
            self._handle_filter_value_changed
        )
        self._class_combo.currentIndexChanged.connect(
            self._handle_class_selection_changed
        )

    def _handle_filter_selection_changed(self, _index: int) -> None:
        self._ensure_required_class_selected()
        self._update_control_states()
        self.filter_changed.emit()

    def _handle_filter_value_changed(self, _value: int) -> None:
        if self.filter_key in self.CONFIDENCE_FILTERS:
            self.filter_changed.emit()

    def _handle_class_selection_changed(self, _index: int) -> None:
        self._ensure_required_class_selected()
        self._update_control_states()
        self.filter_changed.emit()

    def _ensure_required_class_selected(self) -> None:
        if self.filter_key != "missing_class":
            return

        if self.class_id is not None:
            return

        if self._class_combo.count() <= 1:
            return

        self._class_combo.blockSignals(True)

        try:
            self._class_combo.setCurrentIndex(1)
        finally:
            self._class_combo.blockSignals(False)

    def _update_control_states(self) -> None:
        uses_confidence = self.filter_key in self.CONFIDENCE_FILTERS
        class_supported = (
            self.filter_key
            not in self.FILTERS_WITHOUT_CLASS_REFINEMENT
        )

        self._confidence_spin.setEnabled(uses_confidence)
        self._confidence_help.setEnabled(uses_confidence)
        self._class_combo.setEnabled(class_supported)
