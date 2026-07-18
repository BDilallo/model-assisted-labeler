from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.ui.class_panel import ClassPanel
from model_assisted_labeler.ui.image_canvas import ImageCanvas


class MainWindow(QMainWindow):
    """
    Main application window for the model-assisted labeler.

    The window connects user-facing controls to AnnotationController,
    ImageCanvas, and ClassPanel. Annotation data is never edited
    directly from this class.
    """

    WINDOW_TITLE = "Model-Assisted Labeler"

    def __init__(
        self,
        controller: AnnotationController,
    ) -> None:
        super().__init__()

        self._controller = controller

        self._canvas = ImageCanvas(controller)
        self._class_panel = ClassPanel(
            controller=controller,
            canvas=self._canvas,
        )

        self._model_label = QLabel("Model: None")
        self._session_label = QLabel("Session: None")
        self._image_label = QLabel("Image: None")

        self._previous_button = QPushButton("Previous")
        self._next_button = QPushButton("Next")
        self._save_button = QPushButton("Save")
        self._save_next_button = QPushButton("Save && Next")
        self._predict_button = QPushButton("Predict")
        self._fit_button = QPushButton("Fit")

        self._load_model_action = QAction(
            "Load Model...",
            self,
        )
        self._open_session_action = QAction(
            "Open Session...",
            self,
        )
        self._close_session_action = QAction(
            "Close Session",
            self,
        )
        self._save_current_action = QAction(
            "Save Current",
            self,
        )
        self._save_all_action = QAction(
            "Save All Changes",
            self,
        )
        self._save_next_action = QAction(
            "Save and Next",
            self,
        )
        self._previous_action = QAction(
            "Previous Image",
            self,
        )
        self._next_action = QAction(
            "Next Image",
            self,
        )
        self._predict_action = QAction(
            "Predict Current Image",
            self,
        )
        self._replace_predictions_action = QAction(
            "Replace with Predictions",
            self,
        )
        self._clear_annotations_action = QAction(
            "Clear Current Annotations",
            self,
        )
        self._delete_selected_action = QAction(
            "Delete Selected Box",
            self,
        )
        self._fit_action = QAction(
            "Fit Image",
            self,
        )
        self._exit_action = QAction(
            "Exit",
            self,
        )

        self._configure_window()
        self._configure_actions()
        self._build_menu_bar()
        self._build_tool_bar()
        self._build_central_widget()
        self._build_status_bar()
        self._connect_signals()
        self._refresh_interface()

    def closeEvent(
        self,
        event: QCloseEvent,
    ) -> None:
        """
        Confirm unsaved changes before closing the application.
        """
        if not self._confirm_unsaved_changes():
            event.ignore()
            return

        event.accept()

    def _configure_window(self) -> None:
        """
        Configure basic main-window properties.
        """
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

    def _configure_actions(self) -> None:
        """
        Assign shortcuts and descriptive status tips.
        """
        self._load_model_action.setShortcut(
            QKeySequence("Ctrl+M")
        )
        self._load_model_action.setStatusTip(
            "Load an Ultralytics detection model."
        )

        self._open_session_action.setShortcut(
            QKeySequence.StandardKey.Open
        )
        self._open_session_action.setStatusTip(
            "Open image and label directories."
        )

        self._close_session_action.setShortcut(
            QKeySequence("Ctrl+W")
        )

        self._save_current_action.setShortcut(
            QKeySequence.StandardKey.Save
        )
        self._save_current_action.setStatusTip(
            "Save annotations for the current image."
        )

        self._save_all_action.setShortcut(
            QKeySequence("Ctrl+Shift+S")
        )
        self._save_all_action.setStatusTip(
            "Save every image with unsaved changes."
        )

        self._save_next_action.setShortcut(
            QKeySequence("Ctrl+Return")
        )

        self._previous_action.setShortcut(
            QKeySequence("Ctrl+Left")
        )
        self._next_action.setShortcut(
            QKeySequence("Ctrl+Right")
        )

        self._predict_action.setShortcut(
            QKeySequence("P")
        )

        self._delete_selected_action.setShortcut(
            QKeySequence.StandardKey.Delete
        )

        self._fit_action.setShortcut(
            QKeySequence("F")
        )

        self._exit_action.setShortcut(
            QKeySequence.StandardKey.Quit
        )

    def _build_menu_bar(self) -> None:
        """
        Create the application menus.
        """
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self._load_model_action)
        file_menu.addAction(self._open_session_action)
        file_menu.addAction(self._close_session_action)
        file_menu.addSeparator()
        file_menu.addAction(self._save_current_action)
        file_menu.addAction(self._save_all_action)
        file_menu.addAction(self._save_next_action)
        file_menu.addSeparator()
        file_menu.addAction(self._exit_action)

        navigate_menu = self.menuBar().addMenu("Navigate")
        navigate_menu.addAction(self._previous_action)
        navigate_menu.addAction(self._next_action)
        navigate_menu.addAction(self._fit_action)

        annotation_menu = self.menuBar().addMenu("Annotations")
        annotation_menu.addAction(self._predict_action)
        annotation_menu.addAction(
            self._replace_predictions_action
        )
        annotation_menu.addSeparator()
        annotation_menu.addAction(
            self._delete_selected_action
        )
        annotation_menu.addAction(
            self._clear_annotations_action
        )

    def _build_tool_bar(self) -> None:
        """
        Create a compact toolbar for common actions.
        """
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)

        toolbar.addAction(self._load_model_action)
        toolbar.addAction(self._open_session_action)
        toolbar.addSeparator()
        toolbar.addAction(self._save_current_action)
        toolbar.addAction(self._save_all_action)
        toolbar.addSeparator()
        toolbar.addAction(self._predict_action)
        toolbar.addAction(self._fit_action)

        self.addToolBar(toolbar)

    def _build_central_widget(self) -> None:
        """
        Arrange session information, canvas, class panel, and controls.
        """
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        information_layout = QVBoxLayout()
        information_layout.addWidget(self._model_label)
        information_layout.addWidget(self._session_label)
        information_layout.addWidget(self._image_label)

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._class_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1000, 260])

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self._previous_button)
        controls_layout.addWidget(self._next_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._predict_button)
        controls_layout.addWidget(self._fit_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._save_button)
        controls_layout.addWidget(self._save_next_button)

        main_layout.addLayout(information_layout)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addLayout(controls_layout)

        self.setCentralWidget(central_widget)

    def _build_status_bar(self) -> None:
        """
        Create and install the application status bar.
        """
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        status_bar.showMessage(
            "Load a model to begin."
        )

    def _connect_signals(self) -> None:
        """
        Connect actions, buttons, and child-widget signals.
        """
        self._load_model_action.triggered.connect(
            self._load_model
        )
        self._open_session_action.triggered.connect(
            self._open_session
        )
        self._close_session_action.triggered.connect(
            self._close_session
        )
        self._save_current_action.triggered.connect(
            self._save_current_image
        )
        self._save_all_action.triggered.connect(
            self._save_all_changes
        )
        self._save_next_action.triggered.connect(
            self._save_and_next
        )
        self._previous_action.triggered.connect(
            self._previous_image
        )
        self._next_action.triggered.connect(
            self._next_image
        )
        self._predict_action.triggered.connect(
            self._predict_current_image
        )
        self._replace_predictions_action.triggered.connect(
            self._replace_with_predictions
        )
        self._clear_annotations_action.triggered.connect(
            self._clear_current_annotations
        )
        self._delete_selected_action.triggered.connect(
            self._delete_selected_annotation
        )
        self._fit_action.triggered.connect(
            self._canvas.fit_to_image
        )
        self._exit_action.triggered.connect(
            self.close
        )

        self._previous_button.clicked.connect(
            self._previous_image
        )
        self._next_button.clicked.connect(
            self._next_image
        )
        self._save_button.clicked.connect(
            self._save_current_image
        )
        self._save_next_button.clicked.connect(
            self._save_and_next
        )
        self._predict_button.clicked.connect(
            self._predict_current_image
        )
        self._fit_button.clicked.connect(
            self._canvas.fit_to_image
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
        self._canvas.error_occurred.connect(
            self._show_error
        )

        self._class_panel.error_occurred.connect(
            self._show_error
        )

    def _load_model(self) -> None:
        """
        Ask the user for a model file and load it.

        Loading a different model closes the current annotation
        session because the model defines the session's class mapping.
        """
        model_filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Detection Model",
            "",
            (
                "Detection Models (*.pt *.onnx *.engine);;"
                "All Files (*)"
            ),
        )

        if not model_filename:
            return

        if self._controller.has_session:
            if not self._confirm_unsaved_changes():
                return

            try:
                self._controller.close_session(
                    discard_unsaved_changes=True
                )
            except Exception as error:
                self._show_error(str(error))
                return

            self._canvas.clear()
            self._class_panel.clear()

        try:
            self.statusBar().showMessage(
                "Loading detection model..."
            )

            self._controller.load_model(
                Path(model_filename)
            )

        except Exception as error:
            self._show_error(str(error))
            self._refresh_interface()
            return

        self.statusBar().showMessage(
            f"Loaded model: {Path(model_filename).name}",
            5000,
        )

        self._refresh_interface()

    def _open_session(self) -> None:
        """
        Select image and label directories and open a session.
        """
        if not self._controller.model_is_loaded:
            self._show_error(
                "Load a detection model before opening a session."
            )
            return

        if not self._confirm_unsaved_changes():
            return

        image_directory_name = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
        )

        if not image_directory_name:
            return

        label_directory_name = QFileDialog.getExistingDirectory(
            self,
            "Select Label Directory",
        )

        if not label_directory_name:
            return

        recursive_choice = QMessageBox.question(
            self,
            "Search Subdirectories",
            (
                "Should image discovery include subdirectories?"
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
            QMessageBox.StandardButton.No,
        )

        recursive = (
            recursive_choice
            == QMessageBox.StandardButton.Yes
        )

        try:
            self.statusBar().showMessage(
                "Opening annotation session..."
            )

            self._controller.open_session(
                image_directory=Path(
                    image_directory_name
                ),
                label_directory=Path(
                    label_directory_name
                ),
                classes=None,
                recursive=recursive,
            )

            self._class_panel.refresh_classes()
            self._canvas.display_current_image()

        except Exception as error:
            self._show_error(str(error))
            self._refresh_interface()
            return

        session = self._controller.session

        if session is not None and not session.has_images:
            self.statusBar().showMessage(
                "Session opened, but no supported images were found.",
                7000,
            )
        else:
            self.statusBar().showMessage(
                "Annotation session opened.",
                5000,
            )

        self._refresh_interface()

    def _close_session(self) -> None:
        """
        Close the active annotation session.
        """
        if not self._controller.has_session:
            return

        if not self._confirm_unsaved_changes():
            return

        try:
            self._controller.close_session(
                discard_unsaved_changes=True
            )

        except Exception as error:
            self._show_error(str(error))
            return

        self._canvas.clear()
        self._class_panel.clear()

        self.statusBar().showMessage(
            "Annotation session closed.",
            5000,
        )

        self._refresh_interface()

    def _save_current_image(self) -> None:
        """
        Save annotations for the current image.
        """
        try:
            self._controller.save_current_image()

        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            "Current annotations saved.",
            4000,
        )

        self._refresh_interface()

    def _save_all_changes(self) -> None:
        """
        Save every dirty image in the current session.
        """
        try:
            saved_count = (
                self._controller.save_all_changes()
            )

        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            f"Saved {saved_count} label file(s).",
            5000,
        )

        self._refresh_interface()

    def _save_and_next(self) -> None:
        """
        Save the current image and move to the next one.
        """
        try:
            current_image = self._controller.current_image

            if current_image is None:
                return

            session = self._controller.session
            was_last_image = (
                session is not None
                and session.is_last_image
            )

            self._controller.save_and_next()
            self._canvas.display_current_image()

        except Exception as error:
            self._show_error(str(error))
            return

        if was_last_image:
            self.statusBar().showMessage(
                "Current image saved. Already at the final image.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                "Current image saved.",
                3000,
            )

        self._refresh_interface()

    def _previous_image(self) -> None:
        """
        Move to the previous image without automatically saving.
        """
        try:
            self._controller.previous_image()
            self._canvas.display_current_image()

        except Exception as error:
            self._show_error(str(error))
            return

        self._refresh_interface()

    def _next_image(self) -> None:
        """
        Move to the next image without automatically saving.
        """
        try:
            self._controller.next_image()
            self._canvas.display_current_image()

        except Exception as error:
            self._show_error(str(error))
            return

        self._refresh_interface()

    def _predict_current_image(self) -> None:
        """
        Add model predictions while preserving manual and edited boxes.
        """
        try:
            self.statusBar().showMessage(
                "Running model prediction..."
            )

            predictions = (
                self._controller.predict_current_image()
            )

            self._canvas.refresh_annotations()

        except Exception as error:
            self._show_error(str(error))
            self._refresh_interface()
            return

        self.statusBar().showMessage(
            f"Added {len(predictions)} prediction(s).",
            5000,
        )

        self._refresh_interface()

    def _replace_with_predictions(self) -> None:
        """
        Replace every current annotation with model predictions.
        """
        image_record = self._controller.current_image

        if image_record is None:
            return

        confirmation = QMessageBox.warning(
            self,
            "Replace Annotations",
            (
                "This will remove every current annotation from "
                f"'{image_record.filename}' and replace them with "
                "new model predictions. Continue?"
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        try:
            self.statusBar().showMessage(
                "Replacing annotations with predictions..."
            )

            predictions = (
                self._controller
                .replace_annotations_with_predictions()
            )

            self._canvas.refresh_annotations()

        except Exception as error:
            self._show_error(str(error))
            self._refresh_interface()
            return

        self.statusBar().showMessage(
            (
                "Replaced annotations with "
                f"{len(predictions)} prediction(s)."
            ),
            5000,
        )

        self._refresh_interface()

    def _clear_current_annotations(self) -> None:
        """
        Remove all annotations from the current image after confirmation.
        """
        image_record = self._controller.current_image

        if image_record is None:
            return

        if not image_record.annotations:
            return

        confirmation = QMessageBox.question(
            self,
            "Clear Annotations",
            (
                "Remove every annotation from "
                f"'{image_record.filename}'?"
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )

        if confirmation != QMessageBox.StandardButton.Yes:
            return

        try:
            self._controller.clear_current_annotations()
            self._canvas.refresh_annotations()

        except Exception as error:
            self._show_error(str(error))
            return

        self.statusBar().showMessage(
            "Current annotations cleared.",
            4000,
        )

        self._refresh_interface()

    def _delete_selected_annotation(self) -> None:
        """
        Delete the box selected on the canvas.
        """
        if self._canvas.delete_selected_annotation():
            self.statusBar().showMessage(
                "Selected annotation deleted.",
                3000,
            )

        self._refresh_interface()

    def _handle_annotation_change(
        self,
        annotation_index: int,
    ) -> None:
        """
        Refresh labels and controls after an annotation changes.
        """
        del annotation_index
        self._refresh_interface()

    def _refresh_interface(self) -> None:
        """
        Refresh labels and enable controls based on application state.
        """
        has_model = self._controller.model_is_loaded
        has_session = self._controller.has_session
        image_record = self._controller.current_image
        has_image = image_record is not None
        has_selection = (
            self._canvas.selected_annotation_index
            is not None
        )

        model_path = self._controller.model_path

        if model_path is None:
            self._model_label.setText("Model: None")
        else:
            self._model_label.setText(
                f"Model: {model_path.name}"
            )

        session = self._controller.session

        if session is None:
            self._session_label.setText(
                "Session: None"
            )
            self._image_label.setText(
                "Image: None"
            )
        else:
            self._session_label.setText(
                (
                    f"Images: {session.image_directory} | "
                    f"Labels: {session.label_directory}"
                )
            )

            if image_record is None:
                self._image_label.setText(
                    "Image: No supported images found"
                )
            else:
                dirty_marker = (
                    " *unsaved*"
                    if image_record.is_dirty
                    else ""
                )

                self._image_label.setText(
                    (
                        f"Image: {image_record.filename} "
                        f"({session.current_position}/"
                        f"{session.image_count}) | "
                        f"Boxes: {image_record.annotation_count}"
                        f"{dirty_marker}"
                    )
                )

        self._open_session_action.setEnabled(has_model)
        self._close_session_action.setEnabled(has_session)

        self._save_current_action.setEnabled(has_image)
        self._save_all_action.setEnabled(
            has_session
            and self._controller.has_unsaved_changes()
        )
        self._save_next_action.setEnabled(has_image)

        self._previous_action.setEnabled(
            has_image
            and session is not None
            and not session.is_first_image
        )
        self._next_action.setEnabled(
            has_image
            and session is not None
            and not session.is_last_image
        )

        self._predict_action.setEnabled(
            has_model and has_image
        )
        self._replace_predictions_action.setEnabled(
            has_model and has_image
        )
        self._clear_annotations_action.setEnabled(
            has_image
            and bool(image_record.annotations)
        )
        self._delete_selected_action.setEnabled(
            has_selection
        )
        self._fit_action.setEnabled(
            self._canvas.has_image
        )

        self._previous_button.setEnabled(
            self._previous_action.isEnabled()
        )
        self._next_button.setEnabled(
            self._next_action.isEnabled()
        )
        self._save_button.setEnabled(
            self._save_current_action.isEnabled()
        )
        self._save_next_button.setEnabled(
            self._save_next_action.isEnabled()
        )
        self._predict_button.setEnabled(
            self._predict_action.isEnabled()
        )
        self._fit_button.setEnabled(
            self._fit_action.isEnabled()
        )

        if has_session:
            title_suffix = ""

            if self._controller.has_unsaved_changes():
                title_suffix = " *"

            self.setWindowTitle(
                self.WINDOW_TITLE + title_suffix
            )
        else:
            self.setWindowTitle(
                self.WINDOW_TITLE
            )

    def _confirm_unsaved_changes(self) -> bool:
        """
        Ask whether unsaved changes should be saved or discarded.

        Returns True when the requested operation may continue.
        """
        if not self._controller.has_unsaved_changes():
            return True

        choice = QMessageBox.warning(
            self,
            "Unsaved Changes",
            (
                "The current annotation session contains unsaved "
                "changes."
            ),
            (
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            ),
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

        self._refresh_interface()
        return True

    def _show_error(
        self,
        message: str,
    ) -> None:
        """
        Display an error dialog and update the status bar.
        """
        cleaned_message = message.strip()

        if not cleaned_message:
            cleaned_message = "An unknown error occurred."

        QMessageBox.critical(
            self,
            "Error",
            cleaned_message,
        )

        self.statusBar().showMessage(
            cleaned_message,
            8000,
        )
