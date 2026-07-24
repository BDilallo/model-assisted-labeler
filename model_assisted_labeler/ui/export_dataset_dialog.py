from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QThread, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.services.dataset_exporter import (
    DatasetExportCancelled,
    DatasetExportRequest,
    DatasetExportResult,
    DatasetExporter,
)


class _DatasetExportWorker(QObject):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        controller: AnnotationController,
        request: DatasetExportRequest,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._request = request
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request cancellation from the UI thread."""
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = self._controller.export_dataset(
                request=self._request,
                progress_callback=self.progress.emit,
                cancellation_check=self._cancel_event.is_set,
            )
        except DatasetExportCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)
        finally:
            self.finished.emit()


class ExportDatasetDialog(QDialog):
    """Collect options and export the current annotation pool as YOLO."""

    def __init__(
        self,
        controller: AnnotationController,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._thread: QThread | None = None
        self._worker: _DatasetExportWorker | None = None
        self._progress_dialog: QProgressDialog | None = None

        session = controller.session

        if session is None:
            raise RuntimeError("No annotation session is currently open.")

        self._dataset_name_edit = QLineEdit(
            f"{session.name} Dataset"
        )
        self._destination_edit = QLineEdit(
            str(controller.default_dataset_export_directory)
        )
        self._destination_edit.setReadOnly(True)

        self._train_spin = self._percentage_spin_box(80)
        self._validation_spin = self._percentage_spin_box(20)
        self._test_checkbox = QCheckBox("Include test split")
        self._test_spin = self._percentage_spin_box(0)
        self._test_spin.setEnabled(False)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-2_147_483_648, 2_147_483_647)
        self._seed_spin.setValue(42)

        self._total_label = QLabel("Total: 100%")
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)

        self._export_button = QPushButton("Export Dataset")
        self._cancel_button = QPushButton("Cancel")
        self._validation_label = QLabel()
        self._validation_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._validation_label.setStyleSheet(
            "color: #d84a4a; font-size: 11px;"
        )

        self.setWindowTitle("Export Dataset")
        self.setModal(True)
        self.resize(620, 460)
        self.setMinimumWidth(560)

        self._build_layout()
        self._connect_signals()
        self._refresh_validation()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        heading = QLabel("Export Training Dataset")
        heading.setObjectName("exportDatasetHeading")
        layout.addWidget(heading)

        explanation = QLabel(
            "Export every image currently saved in the annotation pool "
            "as a portable Ultralytics YOLO detection dataset. The source "
            "images and session files will not be modified."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        form.addRow("Dataset name:", self._dataset_name_edit)

        destination_row = QHBoxLayout()
        destination_row.addWidget(self._destination_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        destination_row.addWidget(browse_button)
        form.addRow("Export location:", destination_row)

        form.addRow("Train:", self._with_percent_suffix(self._train_spin))
        form.addRow(
            "Validation:",
            self._with_percent_suffix(self._validation_spin),
        )
        form.addRow("", self._test_checkbox)
        form.addRow("Test:", self._with_percent_suffix(self._test_spin))
        form.addRow("", self._total_label)
        form.addRow("Random seed:", self._seed_spin)
        layout.addLayout(form)

        note = QLabel(
            "The same images, percentages, and seed will produce the "
            "same split. Test folders are created only when the test "
            "split is enabled."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(self._summary_label)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._export_button)
        button_row.addWidget(self._cancel_button)
        layout.addLayout(button_row)
        layout.addWidget(self._validation_label)

        browse_button.clicked.connect(self._browse_destination)

    def _connect_signals(self) -> None:
        self._dataset_name_edit.textChanged.connect(
            self._refresh_validation
        )
        self._destination_edit.textChanged.connect(
            self._refresh_validation
        )
        self._train_spin.valueChanged.connect(self._refresh_validation)
        self._validation_spin.valueChanged.connect(
            self._refresh_validation
        )
        self._test_spin.valueChanged.connect(self._refresh_validation)
        self._test_checkbox.toggled.connect(
            self._handle_test_toggled
        )
        self._export_button.clicked.connect(self._start_export)
        self._cancel_button.clicked.connect(self.reject)

    def _browse_destination(self) -> None:
        current_path = Path(self._destination_edit.text())
        starting_directory = (
            current_path
            if current_path.is_dir()
            else current_path.parent
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Export Location",
            str(starting_directory),
        )

        if directory:
            self._destination_edit.setText(directory)

    def _handle_test_toggled(self, checked: bool) -> None:
        self._test_spin.blockSignals(True)
        self._validation_spin.blockSignals(True)
        self._train_spin.blockSignals(True)

        try:
            self._test_spin.setEnabled(checked)

            if checked:
                if self._test_spin.value() == 0:
                    self._test_spin.setValue(10)

                    if self._validation_spin.value() >= 10:
                        self._validation_spin.setValue(
                            self._validation_spin.value() - 10
                        )
                    else:
                        self._train_spin.setValue(
                            max(0, self._train_spin.value() - 10)
                        )
            else:
                self._test_spin.setValue(0)
                self._validation_spin.setValue(
                    max(0, 100 - self._train_spin.value())
                )
        finally:
            self._test_spin.blockSignals(False)
            self._validation_spin.blockSignals(False)
            self._train_spin.blockSignals(False)

        self._refresh_validation()

    def _refresh_validation(self) -> None:
        train = self._train_spin.value()
        validation = self._validation_spin.value()
        test = self._test_spin.value() if self._test_checkbox.isChecked() else 0
        total = train + validation + test
        self._total_label.setText(f"Total: {total}%")

        error_message = ""

        if total != 100:
            error_message = "Split percentages must total 100%."
        elif train <= 0 or validation <= 0:
            error_message = (
                "Train and validation splits must be greater than 0%."
            )
        elif self._test_checkbox.isChecked() and test <= 0:
            error_message = (
                "The test split must be greater than 0% when enabled."
            )
        elif not self._dataset_name_edit.text().strip():
            error_message = "Enter a dataset name."
        elif not self._destination_edit.text().strip():
            error_message = "Choose an export location."

        total_images = self._controller.total_images_annotated
        split_percentages = {
            "train": train,
            "val": validation,
        }

        if self._test_checkbox.isChecked() and test > 0:
            split_percentages["test"] = test

        if (
            not error_message
            and total_images < len(split_percentages)
        ):
            error_message = (
                "Not enough saved images for every enabled split."
            )

        if total == 100:
            split_counts = DatasetExporter.calculate_split_counts(
                total_images=total_images,
                split_percentages=split_percentages,
            )
            summary_parts = [
                f"Saved images: {total_images}",
                f"Train: {split_counts.get('train', 0)}",
                f"Validation: {split_counts.get('val', 0)}",
            ]

            if "test" in split_counts:
                summary_parts.append(f"Test: {split_counts['test']}")

            self._summary_label.setText(" | ".join(summary_parts))
        else:
            self._summary_label.setText(
                f"Saved images available: {total_images}"
            )

        self._validation_label.setText(error_message)
        self._export_button.setEnabled(
            not error_message and total_images > 0
        )

    def _build_request(self, replace_existing: bool) -> DatasetExportRequest:
        return DatasetExportRequest(
            dataset_name=self._dataset_name_edit.text(),
            destination_parent=Path(self._destination_edit.text()),
            train_percent=self._train_spin.value(),
            validation_percent=self._validation_spin.value(),
            test_percent=(
                self._test_spin.value()
                if self._test_checkbox.isChecked()
                else 0
            ),
            random_seed=self._seed_spin.value(),
            replace_existing=replace_existing,
        )

    def _start_export(self) -> None:
        try:
            request = self._build_request(replace_existing=False)
        except Exception as error:
            self._validation_label.setText(str(error))
            self._export_button.setEnabled(False)
            return

        if request.export_directory.exists():
            confirmation = QMessageBox(self)
            confirmation.setWindowTitle("Replace Existing Export")
            confirmation.setIcon(QMessageBox.Icon.Warning)
            confirmation.setText(
                f"A dataset named '{request.dataset_name}' already "
                "exists in the selected location."
            )
            confirmation.setInformativeText(
                "Replace the existing exported dataset? The active "
                "annotation session will not be modified."
            )
            replace_button = confirmation.addButton(
                "Replace Existing Export",
                QMessageBox.ButtonRole.DestructiveRole,
            )
            confirmation.addButton(
                "Choose Another Name",
                QMessageBox.ButtonRole.RejectRole,
            )
            confirmation.exec()

            if confirmation.clickedButton() is not replace_button:
                return

            request = self._build_request(replace_existing=True)

        self._set_form_enabled(False)
        self._progress_dialog = QProgressDialog(
            "Preparing dataset export...",
            "Cancel",
            0,
            100,
            self,
        )
        self._progress_dialog.setWindowTitle("Exporting Dataset")
        self._progress_dialog.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setValue(0)

        self._thread = QThread(self)
        self._worker = _DatasetExportWorker(
            controller=self._controller,
            request=request,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._update_progress)
        self._worker.completed.connect(self._handle_completed)
        self._worker.failed.connect(self._handle_failed)
        self._worker.cancelled.connect(self._handle_cancelled)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._clear_thread_references)
        self._progress_dialog.canceled.connect(self._request_cancel)

        self._thread.start()
        self._progress_dialog.show()

    @Slot(int, int, str)
    def _update_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if self._progress_dialog is None:
            return

        maximum = max(total, 1)
        percentage = round((current / maximum) * 100)
        self._progress_dialog.setLabelText(message)
        self._progress_dialog.setValue(
            min(max(percentage, 0), 100)
        )

    def _request_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

        if self._progress_dialog is not None:
            self._progress_dialog.setLabelText(
                "Cancelling and removing temporary files..."
            )

    @Slot(object)
    def _handle_completed(self, result: DatasetExportResult) -> None:
        self._close_progress_dialog()
        self._show_completion_dialog(result)
        self.accept()

    @Slot(str)
    def _handle_failed(self, message: str) -> None:
        self._close_progress_dialog()
        self._set_form_enabled(True)
        QMessageBox.critical(self, "Dataset Export Failed", message)
        self._refresh_validation()

    @Slot()
    def _handle_cancelled(self) -> None:
        self._close_progress_dialog()
        self._set_form_enabled(True)
        QMessageBox.information(
            self,
            "Export Cancelled",
            "The dataset export was cancelled. Temporary files were removed.",
        )
        self._refresh_validation()

    def _show_completion_dialog(
        self,
        result: DatasetExportResult,
    ) -> None:
        split_lines = [
            f"Train: {result.split_counts.get('train', 0)}",
            f"Validation: {result.split_counts.get('val', 0)}",
        ]

        if "test" in result.split_counts:
            split_lines.append(
                f"Test: {result.split_counts.get('test', 0)}"
            )

        message = QMessageBox(self)
        message.setWindowTitle("Dataset Exported")
        message.setIcon(QMessageBox.Icon.Information)
        message.setText("Dataset exported successfully.")
        message.setInformativeText(
            f"Location: {result.export_directory}\n\n"
            f"Images: {result.total_images}\n"
            f"Annotations: {result.total_annotations}\n"
            + "\n".join(split_lines)
        )

        if result.warnings:
            message.setDetailedText("\n".join(result.warnings))

        open_button = message.addButton(
            "Open Folder",
            QMessageBox.ButtonRole.ActionRole,
        )
        message.addButton("Close", QMessageBox.ButtonRole.AcceptRole)
        message.exec()

        if message.clickedButton() is open_button:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(result.export_directory))
            )

    def _set_form_enabled(self, enabled: bool) -> None:
        self._dataset_name_edit.setEnabled(enabled)
        self._destination_edit.setEnabled(enabled)
        self._train_spin.setEnabled(enabled)
        self._validation_spin.setEnabled(enabled)
        self._test_checkbox.setEnabled(enabled)
        self._test_spin.setEnabled(
            enabled and self._test_checkbox.isChecked()
        )
        self._seed_spin.setEnabled(enabled)
        self._export_button.setEnabled(enabled)
        self._cancel_button.setEnabled(enabled)

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None

    @Slot()
    def _clear_thread_references(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()

        self._thread = None
        self._worker = None

    @staticmethod
    def _percentage_spin_box(value: int) -> QSpinBox:
        spin_box = QSpinBox()
        spin_box.setRange(0, 100)
        spin_box.setValue(value)
        return spin_box

    @staticmethod
    def _with_percent_suffix(spin_box: QSpinBox) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(spin_box)
        row.addWidget(QLabel("%"))
        row.addStretch(1)
        return row
