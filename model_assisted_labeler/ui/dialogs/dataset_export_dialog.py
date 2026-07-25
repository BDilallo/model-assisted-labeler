from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from model_assisted_labeler.services.dataset_export_service import (
    DatasetExportService,
    DatasetExportSettings,
)

_INVALID_FOLDER_NAME_CHARACTERS = set('<>:"/\\|?*')


class DatasetExportDialog(QDialog):
    """Collect the destination, split, and options for a dataset export."""

    def __init__(
        self,
        default_dataset_name: str,
        pooled_image_count: int,
        has_unsaved_changes: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)

        if pooled_image_count < 0:
            raise ValueError("Pooled image count cannot be negative.")

        self._export_service = DatasetExportService()
        self._pooled_image_count = pooled_image_count
        self._result_data: DatasetExportSettings | None = None

        self._output_directory_edit = QLineEdit()
        self._output_directory_edit.setReadOnly(True)
        self._folder_name_edit = QLineEdit(default_dataset_name)

        self._percent_radio = QRadioButton("Percent")
        self._count_radio = QRadioButton("Image count")
        self._percent_radio.setChecked(True)
        self._split_mode_group = QButtonGroup(self)
        self._split_mode_group.addButton(self._percent_radio)
        self._split_mode_group.addButton(self._count_radio)

        self._validation_spin = QSpinBox()
        self._validation_spin.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._preview_label = QLabel()

        self._shuffle_checkbox = QCheckBox("Shuffle images before splitting")
        self._shuffle_checkbox.setChecked(True)
        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText(
            "Random each export (optional integer for a repeatable split)"
        )

        self._remap_checkbox = QCheckBox(
            "Renumber class IDs to be contiguous (0...N-1) in the export"
        )
        self._remap_checkbox.setChecked(True)
        self._remap_checkbox.setToolTip(
            "Session class IDs can have gaps after a class is deleted. "
            "YOLO requires label IDs to directly index the class list, "
            "so this remaps IDs for the exported copy only. The session "
            "itself is never modified."
        )

        self.setWindowTitle("Export Dataset")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._build_layout(has_unsaved_changes)
        self._connect_signals()
        self._set_split_mode(is_percent=True)

    @property
    def result_data(self) -> DatasetExportSettings | None:
        return self._result_data

    def _build_layout(self, has_unsaved_changes: bool) -> None:
        main_layout = QVBoxLayout(self)

        heading = QLabel("Export Dataset")
        heading.setObjectName("datasetExportHeading")
        main_layout.addWidget(heading)

        summary = QLabel(
            f"{self._pooled_image_count} saved image(s) are available "
            "to export."
        )
        main_layout.addWidget(summary)

        if has_unsaved_changes:
            warning = QLabel(
                "Some images have unsaved changes and will not be "
                "included. Save first for a complete export."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b45309;")
            main_layout.addWidget(warning)

        form_layout = QFormLayout()

        output_row = QHBoxLayout()
        output_row.addWidget(self._output_directory_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output_directory)
        output_row.addWidget(browse_button)
        form_layout.addRow("Output location:", output_row)
        form_layout.addRow("Dataset folder name:", self._folder_name_edit)

        split_mode_row = QHBoxLayout()
        split_mode_row.addWidget(self._percent_radio)
        split_mode_row.addWidget(self._count_radio)
        form_layout.addRow("Validation split by:", split_mode_row)
        form_layout.addRow("Validation amount:", self._validation_spin)
        form_layout.addRow("", self._preview_label)

        form_layout.addRow(self._shuffle_checkbox)
        form_layout.addRow("Random seed:", self._seed_edit)
        form_layout.addRow(self._remap_checkbox)

        main_layout.addLayout(form_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        export_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        if export_button is not None:
            export_button.setText("Export")
            export_button.setDefault(True)

        main_layout.addWidget(buttons)

        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

    def _connect_signals(self) -> None:
        self._percent_radio.toggled.connect(
            lambda checked: checked and self._set_split_mode(True)
        )
        self._count_radio.toggled.connect(
            lambda checked: checked and self._set_split_mode(False)
        )
        self._validation_spin.valueChanged.connect(
            self._update_preview
        )

    def _set_split_mode(self, is_percent: bool) -> None:
        self._validation_spin.blockSignals(True)

        if is_percent:
            self._validation_spin.setRange(0, 100)
            self._validation_spin.setSuffix("%")
            self._validation_spin.setValue(20)
        else:
            self._validation_spin.setRange(0, self._pooled_image_count)
            self._validation_spin.setSuffix(" images")
            self._validation_spin.setValue(
                round(self._pooled_image_count * 0.2)
            )

        self._validation_spin.blockSignals(False)
        self._update_preview()

    def _current_settings_for_preview(self) -> DatasetExportSettings:
        return DatasetExportSettings(
            output_directory=Path("."),
            dataset_folder_name="preview",
            validation_is_percent=self._percent_radio.isChecked(),
            validation_amount=self._validation_spin.value(),
        )

    def _update_preview(self) -> None:
        train_count, val_count = self._export_service.compute_split_counts(
            self._pooled_image_count,
            self._current_settings_for_preview(),
        )
        self._preview_label.setText(
            f"Train: {train_count} image(s) | Validation: {val_count} image(s)"
        )

    def _browse_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Output Location",
        )

        if directory:
            self._output_directory_edit.setText(directory)

    def _validate_and_accept(self) -> None:
        if self._pooled_image_count == 0:
            QMessageBox.warning(
                self,
                "Nothing to Export",
                "There are no saved images to export.",
            )
            return

        output_directory_text = self._output_directory_edit.text().strip()

        if not output_directory_text:
            QMessageBox.warning(
                self,
                "Missing Output Location",
                "Select where the dataset folder should be created.",
            )
            return

        output_directory = Path(output_directory_text)

        if not output_directory.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Output Location",
                "The selected output location does not exist.",
            )
            return

        folder_name = self._folder_name_edit.text().strip()

        if not folder_name:
            QMessageBox.warning(
                self,
                "Missing Dataset Folder Name",
                "Enter a name for the dataset folder.",
            )
            return

        if _INVALID_FOLDER_NAME_CHARACTERS.intersection(folder_name):
            QMessageBox.warning(
                self,
                "Invalid Dataset Folder Name",
                'The folder name cannot contain: < > : " / \\ | ? *',
            )
            return

        final_path = output_directory / folder_name

        if final_path.exists() and any(final_path.iterdir()):
            QMessageBox.warning(
                self,
                "Folder Already Exists",
                (
                    f"'{final_path}' already exists and is not empty. "
                    "Choose a different name or location."
                ),
            )
            return

        seed_text = self._seed_edit.text().strip()
        seed: int | None = None

        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Seed",
                    "The random seed must be a whole number.",
                )
                return

        train_count, val_count = self._export_service.compute_split_counts(
            self._pooled_image_count,
            self._current_settings_for_preview(),
        )

        if train_count == 0 or val_count == 0:
            proceed = QMessageBox.question(
                self,
                "Empty Split",
                (
                    f"This split produces {train_count} training and "
                    f"{val_count} validation image(s). Continue anyway?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )

            if proceed != QMessageBox.StandardButton.Yes:
                return

        self._result_data = DatasetExportSettings(
            output_directory=output_directory,
            dataset_folder_name=folder_name,
            validation_is_percent=self._percent_radio.isChecked(),
            validation_amount=self._validation_spin.value(),
            shuffle=self._shuffle_checkbox.isChecked(),
            seed=seed,
            remap_class_ids=self._remap_checkbox.isChecked(),
        )
        self.accept()
