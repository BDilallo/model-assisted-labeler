from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.ui.image_canvas import ImageCanvas


class ClassPanel(QWidget):
    """Display, add, delete, select, and apply session classes."""

    active_class_changed = Signal(int)
    annotation_class_changed = Signal(int, int)
    classes_changed = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        controller: AnnotationController,
        canvas: ImageCanvas,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._canvas = canvas

        self._active_class_id: int | None = None
        self._selected_annotation_index: int | None = None

        self._title_label = QLabel("Classes")
        self._class_list = QListWidget()
        self._active_class_label = QLabel("Drawing class: None")
        self._selected_box_label = QLabel("Selected box: None")
        self._add_class_button = QPushButton("Add Class")
        self._delete_class_button = QPushButton("Delete Class")
        self._apply_class_button = QPushButton(
            "Apply Class to Selected Box"
        )

        self._configure_widgets()
        self._build_layout()
        self._connect_signals()

    @property
    def active_class_id(self) -> int | None:
        return self._active_class_id

    def refresh_classes(self) -> None:
        session = self._controller.session
        previous_class_id = self._active_class_id

        self._class_list.blockSignals(True)

        try:
            self._class_list.clear()
            self._active_class_id = None

            if session is None:
                self._active_class_label.setText("Drawing class: None")
                self._clear_selected_annotation()
                return

            for class_definition in session.classes:
                item = QListWidgetItem(
                    self._format_class_text(
                        class_definition.class_id,
                        class_definition.name,
                    )
                )
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    class_definition.class_id,
                )
                self._class_list.addItem(item)

            if self._class_list.count() == 0:
                self._active_class_label.setText("Drawing class: None")
                self._clear_selected_annotation()
                return

            item_to_select = self._find_class_item(previous_class_id)

            if item_to_select is None:
                item_to_select = self._class_list.item(0)

            self._class_list.setCurrentItem(item_to_select)

        finally:
            self._class_list.blockSignals(False)

        current_item = self._class_list.currentItem()

        if current_item is not None:
            self._activate_item(current_item)

        self._refresh_selected_annotation_display()
        self._update_button_states()

    def set_active_class_id(self, class_id: int) -> None:
        session = self._controller.session

        if session is None or session.get_class(class_id) is None:
            raise ValueError(
                f"Class ID {class_id} is not defined in the session."
            )

        class_item = self._find_class_item(class_id)

        if class_item is None:
            raise ValueError(
                f"Class ID {class_id} is not displayed."
            )

        self._class_list.setCurrentItem(class_item)

    def clear(self) -> None:
        self._class_list.clear()
        self._active_class_id = None
        self._selected_annotation_index = None
        self._active_class_label.setText("Drawing class: None")
        self._selected_box_label.setText("Selected box: None")
        self._update_button_states()

    def _configure_widgets(self) -> None:
        self._title_label.setObjectName("classPanelTitle")
        self._class_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._class_list.setAlternatingRowColors(True)
        self._apply_class_button.setEnabled(False)
        self._apply_class_button.setToolTip(
            "Change the selected box to the active drawing class."
        )

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._class_list, stretch=1)

        class_button_layout = QHBoxLayout()
        class_button_layout.addWidget(self._add_class_button)
        class_button_layout.addWidget(self._delete_class_button)
        layout.addLayout(class_button_layout)

        layout.addWidget(self._active_class_label)
        layout.addWidget(self._selected_box_label)
        layout.addWidget(self._apply_class_button)

    def _connect_signals(self) -> None:
        self._class_list.currentItemChanged.connect(
            self._handle_current_item_changed
        )
        self._add_class_button.clicked.connect(self._add_class)
        self._delete_class_button.clicked.connect(self._delete_class)
        self._apply_class_button.clicked.connect(
            self._apply_active_class_to_selected_box
        )

        self._canvas.annotation_selected.connect(
            self._handle_annotation_selected
        )
        self._canvas.selection_cleared.connect(
            self._clear_selected_annotation
        )
        self._canvas.annotation_deleted.connect(
            self._handle_annotation_deleted
        )
        self._canvas.annotation_updated.connect(
            self._handle_annotation_updated
        )

    def _add_class(self) -> None:
        class_name, accepted = QInputDialog.getText(
            self,
            "Add Class",
            "Class name:",
        )

        if not accepted:
            return

        try:
            class_definition = self._controller.add_class(class_name)
        except Exception as error:
            self.error_occurred.emit(str(error))
            return

        self.refresh_classes()
        self.set_active_class_id(class_definition.class_id)
        self.classes_changed.emit()

    def _delete_class(self) -> None:
        item = self._class_list.currentItem()
        session = self._controller.session

        if item is None or session is None:
            return

        class_id_value = item.data(Qt.ItemDataRole.UserRole)

        if class_id_value is None:
            return

        class_id = int(class_id_value)
        class_definition = session.get_class(class_id)

        if class_definition is None:
            return

        try:
            usage_filenames = self._controller.class_usage_filenames(
                class_id
            )
        except Exception as error:
            self.error_occurred.emit(str(error))
            return

        first_dialog = QMessageBox(self)
        first_dialog.setWindowTitle("Delete Class")
        first_dialog.setIcon(QMessageBox.Icon.Warning)
        first_dialog.setText(
            f"{len(usage_filenames)} Images contain this class. "
            "Do you wish to proceed?"
        )
        proceed_button = first_dialog.addButton(
            "Proceed",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        first_dialog.addButton(
            "Exit",
            QMessageBox.ButtonRole.RejectRole,
        )
        first_dialog.exec()

        if first_dialog.clickedButton() is not proceed_button:
            return

        mode_dialog = QMessageBox(self)
        mode_dialog.setWindowTitle("Class Deletion Behavior")
        mode_dialog.setIcon(QMessageBox.Icon.Warning)
        mode_dialog.setText(
            "Do you wish to remove the class from the annotations or "
            "delete the referenced images and annotations from the pool?"
        )
        remove_button = mode_dialog.addButton(
            "Remove",
            QMessageBox.ButtonRole.ActionRole,
        )
        delete_button = mode_dialog.addButton(
            "Delete",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        mode_dialog.addButton(
            "Exit",
            QMessageBox.ButtonRole.RejectRole,
        )
        mode_dialog.exec()

        clicked_button = mode_dialog.clickedButton()

        if clicked_button is remove_button:
            mode = "remove"
        elif clicked_button is delete_button:
            mode = "delete"
        else:
            return

        typed_name, accepted = QInputDialog.getText(
            self,
            "Final Confirmation",
            (
                f"Type '{class_definition.name}' to confirm class "
                f"{mode}:"
            ),
        )

        if not accepted:
            return

        if typed_name != class_definition.name:
            QMessageBox.warning(
                self,
                "Name Does Not Match",
                "The class name did not match. Nothing was changed.",
            )
            return

        try:
            self._controller.delete_class(class_id, mode)
        except Exception as error:
            self.error_occurred.emit(str(error))
            return

        self._canvas.refresh_annotations()
        self.refresh_classes()
        self.classes_changed.emit()

    def _handle_current_item_changed(
        self,
        current_item: QListWidgetItem | None,
        previous_item: QListWidgetItem | None,
    ) -> None:
        del previous_item

        if current_item is None:
            self._active_class_id = None
            self._active_class_label.setText("Drawing class: None")
            self._update_button_states()
            return

        self._activate_item(current_item)

    def _activate_item(self, item: QListWidgetItem) -> None:
        class_id_value = item.data(Qt.ItemDataRole.UserRole)

        if class_id_value is None:
            self.error_occurred.emit(
                "The selected class does not contain a class ID."
            )
            return

        class_id = int(class_id_value)
        session = self._controller.session

        if session is None:
            self.error_occurred.emit(
                "No annotation session is currently open."
            )
            return

        class_definition = session.get_class(class_id)

        if class_definition is None:
            self.error_occurred.emit(
                f"Class ID {class_id} is not defined."
            )
            return

        try:
            self._canvas.set_active_class_id(class_id)
        except Exception as error:
            self.error_occurred.emit(str(error))
            return

        self._active_class_id = class_id
        self._active_class_label.setText(
            "Drawing class: "
            f"{class_definition.class_id} - {class_definition.name}"
        )
        self._update_button_states()
        self.active_class_changed.emit(class_id)

    def _handle_annotation_selected(self, annotation_index: int) -> None:
        self._selected_annotation_index = annotation_index
        self._refresh_selected_annotation_display()
        self._update_button_states()

    def _clear_selected_annotation(self) -> None:
        self._selected_annotation_index = None
        self._selected_box_label.setText("Selected box: None")
        self._update_button_states()

    def _handle_annotation_deleted(self, deleted_index: int) -> None:
        del deleted_index
        current_selection = self._canvas.selected_annotation_index

        if current_selection is None:
            self._clear_selected_annotation()
        else:
            self._handle_annotation_selected(current_selection)

    def _handle_annotation_updated(self, annotation_index: int) -> None:
        if self._selected_annotation_index == annotation_index:
            self._refresh_selected_annotation_display()

    def _apply_active_class_to_selected_box(self) -> None:
        annotation_index = self._selected_annotation_index
        class_id = self._active_class_id

        if annotation_index is None or class_id is None:
            return

        try:
            changed = self._canvas.change_selected_annotation_class(
                class_id
            )
        except Exception as error:
            self.error_occurred.emit(str(error))
            return

        if changed:
            self._refresh_selected_annotation_display()
            self.annotation_class_changed.emit(
                annotation_index,
                class_id,
            )

    def _refresh_selected_annotation_display(self) -> None:
        annotation_index = self._selected_annotation_index
        image_record = self._controller.current_image
        session = self._controller.session

        if (
            annotation_index is None
            or image_record is None
            or session is None
            or annotation_index < 0
            or annotation_index >= len(image_record.annotations)
        ):
            self._selected_box_label.setText("Selected box: None")
            self._update_button_states()
            return

        annotation = image_record.annotations[annotation_index]
        class_name = session.get_class_name(annotation.class_id) or "Unknown"
        self._selected_box_label.setText(
            f"Selected box {annotation_index + 1}: "
            f"{annotation.class_id} - {class_name}"
        )
        self._update_button_states()

    def _update_button_states(self) -> None:
        has_session = self._controller.has_session
        self._add_class_button.setEnabled(has_session)
        self._delete_class_button.setEnabled(
            has_session and self._class_list.currentItem() is not None
        )
        self._apply_class_button.setEnabled(
            self._active_class_id is not None
            and self._selected_annotation_index is not None
        )

    def _find_class_item(
        self,
        class_id: int | None,
    ) -> QListWidgetItem | None:
        if class_id is None:
            return None

        for item_index in range(self._class_list.count()):
            item = self._class_list.item(item_index)
            item_class_id = item.data(Qt.ItemDataRole.UserRole)

            if item_class_id is not None and int(item_class_id) == class_id:
                return item

        return None

    @staticmethod
    def _format_class_text(class_id: int, class_name: str) -> str:
        return f"{class_id}: {class_name}"
