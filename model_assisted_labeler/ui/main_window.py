from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.ui.batch_auto_annotation_dialog import (
    BatchAutoAnnotationDialog,
)
from model_assisted_labeler.ui.class_panel import ClassPanel
from model_assisted_labeler.ui.image_canvas import ImageCanvas
from model_assisted_labeler.ui.image_filter_bar import ImageFilterBar


class MainWindow(QMainWindow):
    """Main annotation window for an already selected saved session."""

    WINDOW_TITLE = "Model-Assisted Labeler"
    PREFETCH_DELAY_MS = 2000
    PREFETCH_RADIUS = 5

    def __init__(
        self,
        controller: AnnotationController,
    ) -> None:
        super().__init__()
        self._controller = controller

        self._canvas = ImageCanvas(controller)
        self._class_panel = ClassPanel(controller, self._canvas)
        self._filter_bar = ImageFilterBar()
        self._filtered_image_indexes: list[int] = []

        self._session_label = QLabel()
        self._model_label = QLabel()
        self._image_label = QLabel()
        self._pool_label = QLabel()

        self._back_button = QPushButton("Back")
        self._next_button = QPushButton("Next")
        self._predict_button = QPushButton("Predict / Refresh")
        self._auto_predict_button = QPushButton("Auto Predict: Off")
        self._batch_auto_annotate_button = QPushButton(
            "Batch Auto Annotate..."
        )
        self._fit_button = QPushButton("Fit")
        self._save_button = QPushButton("Save")
        self._save_next_button = QPushButton("Save && Next")
        self._remove_pool_button = QPushButton(
            "Remove from Annotation Pool"
        )

        self._save_action = QAction("Save Current", self)
        self._save_all_action = QAction("Save All Changes", self)
        self._save_next_action = QAction("Save and Next", self)
        self._back_action = QAction("Back", self)
        self._next_action = QAction("Next", self)
        self._predict_action = QAction("Predict / Refresh", self)
        self._replace_action = QAction(
            "Replace with Predictions",
            self,
        )
        self._batch_auto_annotate_action = QAction(
            "Batch Auto Annotate...",
            self,
        )
        self._clear_action = QAction("Clear Current Boxes", self)
        self._delete_box_action = QAction("Delete Selected Box", self)
        self._remove_pool_action = QAction(
            "Remove from Annotation Pool",
            self,
        )
        self._fit_action = QAction("Fit Image", self)
        self._exit_action = QAction("Exit", self)

        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.setInterval(self.PREFETCH_DELAY_MS)
        self._navigation_in_progress = False

        self._configure_window()
        self._configure_actions()
        self._configure_buttons()
        self._build_menu_bar()
        self._build_central_widget()
        self._build_status_bar()
        self._connect_signals()

        self._class_panel.refresh_classes()
        self._refresh_filter_classes()
        self._rebuild_filter_indexes()
        self._display_current_image_with_auto_prediction()
        self._refresh_interface()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_unsaved_changes():
            event.ignore()
            return

        event.accept()

    def _configure_window(self) -> None:
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1320, 840)
        self.setMinimumSize(940, 620)

    def _configure_actions(self) -> None:
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_all_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_next_action.setShortcut(QKeySequence("Ctrl+Return"))
        self._back_action.setShortcut(QKeySequence("Ctrl+Left"))
        self._next_action.setShortcut(QKeySequence("Ctrl+Right"))
        self._predict_action.setShortcut(QKeySequence("P"))
        self._delete_box_action.setShortcut(
            QKeySequence.StandardKey.Delete
        )
        self._fit_action.setShortcut(QKeySequence("F"))
        self._exit_action.setShortcut(QKeySequence.StandardKey.Quit)

    def _configure_buttons(self) -> None:
        self._auto_predict_button.setCheckable(True)
        self._auto_predict_button.setToolTip(
            "Predict automatically only when an image has no preserved "
            "or unsaved annotations."
        )
        self._predict_button.setToolTip(
            "Run or refresh prediction for the current image."
        )
        self._batch_auto_annotate_button.setToolTip(
            "Predict and save every clean image that is not already in "
            "the annotation pool."
        )
        self._remove_pool_button.setToolTip(
            "Delete only the session-owned image copy and annotation. "
            "The source image is never modified."
        )

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self._save_action)
        file_menu.addAction(self._save_all_action)
        file_menu.addAction(self._save_next_action)
        file_menu.addSeparator()
        file_menu.addAction(self._exit_action)

        navigate_menu = self.menuBar().addMenu("Navigate")
        navigate_menu.addAction(self._back_action)
        navigate_menu.addAction(self._next_action)
        navigate_menu.addAction(self._fit_action)

        annotation_menu = self.menuBar().addMenu("Annotations")
        annotation_menu.addAction(self._predict_action)
        annotation_menu.addAction(self._replace_action)
        annotation_menu.addAction(self._batch_auto_annotate_action)
        annotation_menu.addSeparator()
        annotation_menu.addAction(self._delete_box_action)
        annotation_menu.addAction(self._clear_action)
        annotation_menu.addSeparator()
        annotation_menu.addAction(self._remove_pool_action)

    def _build_central_widget(self) -> None:
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        information_layout = QVBoxLayout()
        information_layout.addWidget(self._session_label)
        information_layout.addWidget(self._model_label)
        information_layout.addWidget(self._image_label)
        information_layout.addWidget(self._pool_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._class_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1030, 290])

        prediction_layout = QVBoxLayout()
        prediction_layout.setSpacing(4)
        prediction_layout.addWidget(self._predict_button)
        prediction_layout.addWidget(self._auto_predict_button)
        prediction_layout.addWidget(
            self._batch_auto_annotate_button
        )

        save_layout = QVBoxLayout()
        save_layout.setSpacing(4)
        save_layout.addWidget(self._save_button)
        save_layout.addWidget(self._save_next_button)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self._back_button)
        controls_layout.addWidget(self._next_button)
        controls_layout.addStretch(1)
        controls_layout.addLayout(prediction_layout)
        controls_layout.addWidget(self._fit_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._remove_pool_button)
        controls_layout.addLayout(save_layout)

        main_layout.addLayout(information_layout)
        main_layout.addWidget(self._filter_bar)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addLayout(controls_layout)
        self.setCentralWidget(central_widget)

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        status_bar.showMessage(
            "Source images are read-only. All saved data stays inside "
            "the session folder.",
            7000,
        )

    def _connect_signals(self) -> None:
        self._save_action.triggered.connect(self._save_current_image)
        self._save_all_action.triggered.connect(self._save_all_changes)
        self._save_next_action.triggered.connect(self._save_and_next)
        self._back_action.triggered.connect(self._back_image)
        self._next_action.triggered.connect(self._next_image)
        self._predict_action.triggered.connect(
            self._predict_current_image
        )
        self._replace_action.triggered.connect(
            self._replace_with_predictions
        )
        self._batch_auto_annotate_action.triggered.connect(
            self._batch_auto_annotate
        )
        self._clear_action.triggered.connect(
            self._clear_current_annotations
        )
        self._delete_box_action.triggered.connect(
            self._delete_selected_annotation
        )
        self._remove_pool_action.triggered.connect(
            self._remove_from_annotation_pool
        )
        self._fit_action.triggered.connect(self._canvas.fit_to_image)
        self._exit_action.triggered.connect(self.close)

        self._back_button.clicked.connect(self._back_image)
        self._next_button.clicked.connect(self._next_image)
        self._predict_button.clicked.connect(
            self._predict_current_image
        )
        self._auto_predict_button.toggled.connect(
            self._handle_auto_predict_toggled
        )
        self._batch_auto_annotate_button.clicked.connect(
            self._batch_auto_annotate
        )
        self._fit_button.clicked.connect(self._canvas.fit_to_image)
        self._save_button.clicked.connect(self._save_current_image)
        self._save_next_button.clicked.connect(self._save_and_next)
        self._remove_pool_button.clicked.connect(
            self._remove_from_annotation_pool
        )

        self._canvas.annotation_created.connect(
            self._handle_annotation_change
        )
        self._canvas.annotation_updated.connect(
            self._handle_annotation_change
        )
        self._canvas.annotation_deleted.connect(
            self._handle_annotation_change
        )
        self._canvas.annotation_selected.connect(
            self._refresh_interface
        )
        self._canvas.selection_cleared.connect(
            self._refresh_interface
        )
        self._canvas.error_occurred.connect(self._show_error)

        self._class_panel.error_occurred.connect(self._show_error)
        self._class_panel.classes_changed.connect(
            self._handle_classes_changed
        )
        self._filter_bar.filter_changed.connect(
            self._apply_image_filter
        )

        self._prefetch_timer.timeout.connect(
            self._prefetch_nearby_annotations
        )

    def _display_current_image_with_auto_prediction(self) -> None:
        prediction_count: int | None = None

        if (
            self._auto_predict_button.isChecked()
            and self._controller.should_auto_predict_current_image()
        ):
            try:
                predictions = self._run_prediction()
                prediction_count = len(predictions)
                self._refresh_current_filter_membership()
            except Exception as error:
                self._show_error(str(error))

        self._canvas.display_current_image()
        self._restart_prefetch_timer()

        if prediction_count is not None:
            self.statusBar().showMessage(
                f"Auto prediction added {prediction_count} box(es).",
                4000,
            )

    def _back_image(self) -> None:
        target_index = self._previous_matching_index()

        if target_index is None:
            return

        self._navigate_to_filtered_index(target_index)

    def _next_image(self) -> None:
        target_index = self._next_matching_index()

        if target_index is None:
            return

        self._navigate_to_filtered_index(target_index)

    def _save_current_image(self) -> None:
        try:
            self._controller.save_current_image()
            self._refresh_current_filter_membership()
        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            "Image copy and annotation saved to the session pool.",
            4000,
        )
        self._refresh_interface()

    def _save_all_changes(self) -> None:
        try:
            saved_count = self._controller.save_all_changes()
            self._rebuild_filter_indexes()
        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            f"Saved {saved_count} image(s) to the annotation pool.",
            5000,
        )
        self._refresh_interface()

    def _save_and_next(self) -> None:
        """Save now and defer graphics teardown until the click returns."""
        if self._navigation_in_progress:
            return

        session = self._controller.session

        if session is None or session.current_image is None:
            return

        preferred_next_index = self._next_matching_index()

        try:
            self._controller.save_current_image()
            self._refresh_current_filter_membership()
        except Exception as error:
            self._show_error(str(error))
            return

        # Deleting a selected QGraphicsItem hierarchy from inside the
        # QPushButton.clicked call stack can crash PySide on Windows. Let
        # the click event finish before removing the old scene items.
        self._navigation_in_progress = True
        self._refresh_interface()
        QTimer.singleShot(
            0,
            lambda: self._complete_save_and_next(
                preferred_next_index
            ),
        )

    def _complete_save_and_next(
        self,
        preferred_next_index: int | None,
    ) -> None:
        """Complete deferred filtered navigation after Save & Next."""
        try:
            target_index = preferred_next_index

            if target_index not in self._filtered_image_indexes:
                target_index = self._next_matching_index()

            if target_index is None:
                self.statusBar().showMessage(
                    "Image saved. No later image matches the filter.",
                    4000,
                )
            else:
                self._prepare_canvas_for_navigation()
                self._controller.go_to_image(target_index)
                self._display_current_image_with_auto_prediction()
                self.statusBar().showMessage(
                    "Image saved and next matching image loaded.",
                    3000,
                )

        except Exception as error:
            self._show_error(str(error))

        finally:
            self._navigation_in_progress = False
            self._refresh_interface()

    def _remove_from_annotation_pool(self) -> None:
        image_record = self._controller.current_image

        if image_record is None or not image_record.in_annotation_pool:
            return

        confirmation = QMessageBox.warning(
            self,
            "Remove from Annotation Pool",
            (
                f"Remove '{image_record.filename}' and its annotation "
                "from this session's dataset?\n\n"
                "The original source image will not be modified."
            ),
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        preferred_next_index = self._next_matching_index()

        try:
            self._controller.remove_current_from_annotation_pool()
            self._refresh_current_filter_membership()

            target_index = preferred_next_index

            if target_index not in self._filtered_image_indexes:
                target_index = self._next_matching_index()

            if target_index is not None:
                self._prepare_canvas_for_navigation()
                self._controller.go_to_image(target_index)
                self._display_current_image_with_auto_prediction()
            else:
                self._canvas.display_current_image()
                self._restart_prefetch_timer()

        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            "Removed from the annotation pool. Source image unchanged.",
            5000,
        )
        self._refresh_interface()

    def _batch_auto_annotate(self) -> None:
        candidate_count = (
            self._controller.batch_auto_annotation_candidate_count()
        )

        if candidate_count == 0:
            QMessageBox.information(
                self,
                "Batch Auto Annotate",
                (
                    "No clean, unsaved images are available for batch "
                    "annotation."
                ),
            )
            return

        dialog = BatchAutoAnnotationDialog(
            image_count=candidate_count,
            parent=self,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        progress_dialog = QProgressDialog(
            "Preparing batch annotation...",
            "Cancel",
            0,
            candidate_count,
            self,
        )
        progress_dialog.setWindowTitle("Batch Auto Annotate")
        progress_dialog.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setValue(0)

        self._prefetch_timer.stop()
        QApplication.processEvents()

        def update_progress(
            completed: int,
            total: int,
            filename: str,
        ) -> None:
            progress_dialog.setMaximum(total)
            progress_dialog.setLabelText(
                f"Processed {completed} of {total}: {filename}"
            )
            progress_dialog.setValue(completed)
            QApplication.processEvents()

        try:
            result = self._controller.batch_auto_annotate(
                confidence_threshold=dialog.confidence_threshold,
                progress_callback=update_progress,
                cancellation_requested=(
                    progress_dialog.wasCanceled
                ),
            )
        except Exception as error:
            progress_dialog.close()
            self._canvas.refresh_annotations()
            self._rebuild_filter_indexes()
            self._restart_prefetch_timer()
            self._refresh_interface()
            self._show_error(str(error))
            return

        progress_dialog.close()
        self._canvas.refresh_annotations()
        self._rebuild_filter_indexes()
        self._ensure_current_filter_visibility()
        self._restart_prefetch_timer()
        self._refresh_interface()

        status = "cancelled" if result.cancelled else "complete"
        summary = (
            f"Batch auto annotation {status}.\n\n"
            f"Processed: {result.processed_images} of "
            f"{result.candidate_images}\n"
            f"Saved: {result.saved_images}\n"
            "Left unsaved because no qualifying boxes were found: "
            f"{result.rejected_images}"
        )
        QMessageBox.information(
            self,
            "Batch Auto Annotate",
            summary,
        )
        self.statusBar().showMessage(
            f"Batch annotation saved {result.saved_images} image(s).",
            6000,
        )

    def _predict_current_image(self) -> None:
        try:
            predictions = self._run_prediction()
            self._canvas.refresh_annotations()
            self._refresh_current_filter_membership()
        except Exception as error:
            self._show_error(str(error))
            self._refresh_interface()
            return

        self.statusBar().showMessage(
            f"Prediction refreshed with {len(predictions)} box(es).",
            5000,
        )
        self._refresh_interface()

    def _run_prediction(self) -> list:
        self.statusBar().showMessage("Running model prediction...")
        QApplication.processEvents()
        return self._controller.predict_current_image()

    def _replace_with_predictions(self) -> None:
        image_record = self._controller.current_image

        if image_record is None:
            return

        confirmation = QMessageBox.warning(
            self,
            "Replace Annotations",
            (
                "Replace every current box with new model predictions "
                f"for '{image_record.filename}'?"
            ),
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        try:
            predictions = (
                self._controller.replace_annotations_with_predictions()
            )
            self._canvas.refresh_annotations()
            self._refresh_current_filter_membership()
        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            f"Replaced with {len(predictions)} prediction(s).",
            5000,
        )
        self._refresh_interface()

    def _clear_current_annotations(self) -> None:
        image_record = self._controller.current_image

        if image_record is None or not image_record.annotations:
            return

        confirmation = QMessageBox.question(
            self,
            "Clear Boxes",
            f"Remove every box from '{image_record.filename}'?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        self._controller.clear_current_annotations()
        self._canvas.refresh_annotations()
        self._refresh_current_filter_membership()
        self._refresh_interface()

    def _delete_selected_annotation(self) -> None:
        if self._canvas.delete_selected_annotation():
            self._refresh_current_filter_membership()
            self.statusBar().showMessage(
                "Selected box deleted.",
                3000,
            )

        self._refresh_interface()

    def _handle_auto_predict_toggled(self, checked: bool) -> None:
        self._auto_predict_button.setText(
            "Auto Predict: On" if checked else "Auto Predict: Off"
        )

        if checked:
            if self._controller.should_auto_predict_current_image():
                self._predict_current_image()
            else:
                self.statusBar().showMessage(
                    "Automatic prediction enabled. Preserved or existing "
                    "annotations will not be replaced.",
                    5000,
                )
        else:
            self.statusBar().showMessage(
                "Automatic prediction disabled.",
                3000,
            )

        self._refresh_interface()

    def _handle_annotation_change(self, annotation_index: int) -> None:
        del annotation_index
        self._refresh_current_filter_membership()
        self._refresh_interface()

    def _handle_classes_changed(self) -> None:
        self._canvas.refresh_annotations()
        self._refresh_filter_classes()
        self._apply_image_filter()

    def _refresh_filter_classes(self) -> None:
        session = self._controller.session
        classes = list(session.classes) if session is not None else []
        self._filter_bar.set_classes(classes)

    def _rebuild_filter_indexes(self) -> None:
        session = self._controller.session

        if session is None:
            self._filtered_image_indexes = []
            self._filter_bar.set_result_count(0, 0)
            return

        self._filtered_image_indexes = (
            self._controller.image_indexes_matching(
                filter_key=self._filter_bar.filter_key,
                confidence_threshold=(
                    self._filter_bar.confidence_threshold
                ),
                class_id=self._filter_bar.class_id,
            )
        )
        self._filter_bar.set_result_count(
            len(self._filtered_image_indexes),
            session.image_count,
        )

    def _refresh_current_filter_membership(self) -> None:
        session = self._controller.session

        if session is None or not session.images:
            self._filtered_image_indexes = []
            self._filter_bar.set_result_count(0, 0)
            return

        current_index = session.current_index
        matches = self._controller.image_index_matches_filter(
            image_index=current_index,
            filter_key=self._filter_bar.filter_key,
            confidence_threshold=self._filter_bar.confidence_threshold,
            class_id=self._filter_bar.class_id,
        )
        currently_in_filter = current_index in self._filtered_image_indexes

        if matches and not currently_in_filter:
            self._filtered_image_indexes.append(current_index)
            self._filtered_image_indexes.sort()
        elif not matches and currently_in_filter:
            self._filtered_image_indexes.remove(current_index)

        self._filter_bar.set_result_count(
            len(self._filtered_image_indexes),
            session.image_count,
        )

    def _apply_image_filter(self) -> None:
        try:
            self._rebuild_filter_indexes()
            self._ensure_current_filter_visibility()
        except Exception as error:
            self._show_error(str(error))
            return

        self._refresh_interface()

    def _ensure_current_filter_visibility(self) -> None:
        session = self._controller.session

        if session is None:
            return

        if not self._filtered_image_indexes:
            if self._canvas.has_image:
                self._prepare_canvas_for_navigation()

            self.statusBar().showMessage(
                "No images match the selected filter.",
                4000,
            )
            return

        current_index = session.current_index

        if current_index in self._filtered_image_indexes:
            if not self._canvas.has_image:
                self._controller.go_to_image(current_index)
                self._display_current_image_with_auto_prediction()
            return

        target_index = next(
            (
                index
                for index in self._filtered_image_indexes
                if index > current_index
            ),
            self._filtered_image_indexes[-1],
        )
        self._prepare_canvas_for_navigation()
        self._controller.go_to_image(target_index)
        self._display_current_image_with_auto_prediction()

    def _navigate_to_filtered_index(self, target_index: int) -> None:
        if target_index not in self._filtered_image_indexes:
            return

        try:
            self._prepare_canvas_for_navigation()
            self._controller.go_to_image(target_index)
            self._display_current_image_with_auto_prediction()
        except Exception as error:
            self._show_error(str(error))
            return

        self._refresh_interface()

    def _previous_matching_index(self) -> int | None:
        session = self._controller.session

        if session is None:
            return None

        previous_indexes = [
            index
            for index in self._filtered_image_indexes
            if index < session.current_index
        ]

        if not previous_indexes:
            return None

        return previous_indexes[-1]

    def _next_matching_index(self) -> int | None:
        session = self._controller.session

        if session is None:
            return None

        return next(
            (
                index
                for index in self._filtered_image_indexes
                if index > session.current_index
            ),
            None,
        )

    def _filtered_current_position(self) -> int | None:
        session = self._controller.session

        if session is None:
            return None

        try:
            return (
                self._filtered_image_indexes.index(
                    session.current_index
                )
                + 1
            )
        except ValueError:
            return None

    def _prepare_canvas_for_navigation(self) -> None:
        """Safely release graphics items before changing images."""
        self._prefetch_timer.stop()
        self._canvas.clear()

    def _restart_prefetch_timer(self) -> None:
        self._prefetch_timer.start()

    def _prefetch_nearby_annotations(self) -> None:
        try:
            self._controller.prefetch_nearby_annotations(
                self.PREFETCH_RADIUS
            )
        except Exception as error:
            self.statusBar().showMessage(str(error), 5000)

    def _refresh_interface(self) -> None:
        session = self._controller.session
        definition = self._controller.session_definition
        image_record = self._controller.current_image
        has_image = bool(image_record and self._canvas.has_image)
        has_boxes = bool(
            has_image and image_record and image_record.annotations
        )
        has_selection = (
            self._canvas.selected_annotation_index is not None
        )
        has_model = self._controller.model_is_loaded
        batch_candidate_count = (
            self._controller.batch_auto_annotation_candidate_count()
        )

        if session is None or definition is None:
            self._session_label.setText("Session: None")
            self._model_label.setText("Model: None")
            self._image_label.setText("Image: None")
            self._pool_label.setText("Annotated images: 0")
            self._filter_bar.set_result_count(0, 0)
            return

        self._session_label.setText(
            f"Session: {session.name} | Source: {session.image_directory}"
        )
        model_path = self._controller.model_path
        self._model_label.setText(
            f"Model: {model_path.name if model_path else 'None'}"
        )

        if session.image_count == 0:
            self._image_label.setText(
                "Image: No supported top-level images found"
            )
        elif not has_image and not self._filtered_image_indexes:
            self._image_label.setText(
                "Image: No images match the selected filter"
            )
        elif image_record is None:
            self._image_label.setText("Image: None")
        else:
            dirty_marker = " *unsaved*" if image_record.is_dirty else ""
            pool_marker = (
                "in annotation pool"
                if image_record.in_annotation_pool
                else "not saved"
            )
            filtered_position = self._filtered_current_position()

            if self._filter_bar.is_all_images:
                position_text = (
                    f"{session.current_position}/{session.image_count}"
                )
            elif filtered_position is None:
                position_text = (
                    "current image no longer matches filter | "
                    f"{session.current_position}/{session.image_count} total"
                )
            else:
                position_text = (
                    f"{filtered_position}/"
                    f"{len(self._filtered_image_indexes)} filtered | "
                    f"{session.current_position}/{session.image_count} total"
                )

            self._image_label.setText(
                f"Image: {image_record.filename} ({position_text}) | "
                f"Boxes: {image_record.annotation_count} | "
                f"{pool_marker}{dirty_marker}"
            )

        self._pool_label.setText(
            "Total images annotated: "
            f"{self._controller.total_images_annotated}"
        )

        interaction_enabled = not self._navigation_in_progress
        can_go_back = bool(
            interaction_enabled
            and has_image
            and self._previous_matching_index() is not None
        )
        can_go_next = bool(
            interaction_enabled
            and has_image
            and self._next_matching_index() is not None
        )
        can_save = interaction_enabled and has_image and has_boxes
        can_remove_pool = bool(
            interaction_enabled
            and has_image
            and image_record
            and image_record.in_annotation_pool
        )

        self._save_action.setEnabled(can_save)
        self._save_all_action.setEnabled(
            self._controller.has_unsaved_changes()
        )
        self._save_next_action.setEnabled(can_save)
        self._back_action.setEnabled(can_go_back)
        self._next_action.setEnabled(can_go_next)
        self._predict_action.setEnabled(
            interaction_enabled and has_model and has_image
        )
        self._replace_action.setEnabled(
            interaction_enabled and has_model and has_image
        )
        self._batch_auto_annotate_action.setEnabled(
            interaction_enabled
            and has_model
            and batch_candidate_count > 0
        )
        self._clear_action.setEnabled(interaction_enabled and has_boxes)
        self._delete_box_action.setEnabled(
            interaction_enabled and has_selection
        )
        self._remove_pool_action.setEnabled(can_remove_pool)
        self._fit_action.setEnabled(self._canvas.has_image)

        self._back_button.setEnabled(can_go_back)
        self._next_button.setEnabled(can_go_next)
        self._predict_button.setEnabled(
            interaction_enabled and has_model and has_image
        )
        self._auto_predict_button.setEnabled(
            interaction_enabled and has_model and has_image
        )
        self._batch_auto_annotate_button.setEnabled(
            interaction_enabled
            and has_model
            and batch_candidate_count > 0
        )
        self._fit_button.setEnabled(
            interaction_enabled and self._canvas.has_image
        )
        self._save_button.setEnabled(can_save)
        self._save_next_button.setEnabled(can_save)
        self._remove_pool_button.setEnabled(can_remove_pool)

        title_suffix = " *" if self._controller.has_unsaved_changes() else ""
        self.setWindowTitle(
            f"{self.WINDOW_TITLE} - {session.name}{title_suffix}"
        )

    def _confirm_unsaved_changes(self) -> bool:
        if not self._controller.has_unsaved_changes():
            return True

        choice = QMessageBox.warning(
            self,
            "Unsaved Changes",
            (
                "The session contains unsaved annotation changes. Save "
                "them before closing?"
            ),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        if choice == QMessageBox.StandardButton.Cancel:
            return False

        if choice == QMessageBox.StandardButton.Discard:
            return True

        try:
            self._controller.save_all_changes()
        except Exception as error:
            self._show_error(str(error))
            return False

        return True

    def _show_error(self, message: str) -> None:
        cleaned_message = message.strip() or "An unknown error occurred."
        QMessageBox.critical(self, "Error", cleaned_message)
        self.statusBar().showMessage(cleaned_message, 8000)
