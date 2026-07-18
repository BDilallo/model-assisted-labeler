from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.ui.image_canvas import ImageCanvas


class ClassPanel(QWidget):
    """
    Displays the annotation classes available in the current session.

    Selecting a class changes the active class used when drawing new
    bounding boxes. The selected class can also be explicitly applied
    to the currently selected annotation.
    """

    active_class_changed = Signal(int)
    annotation_class_changed = Signal(int, int)
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

        self._active_class_label = QLabel(
            "Drawing class: None"
        )

        self._selected_box_label = QLabel(
            "Selected box: None"
        )

        self._apply_class_button = QPushButton(
            "Apply Class to Selected Box"
        )

        self._configure_widgets()
        self._build_layout()
        self._connect_signals()

    @property
    def active_class_id(self) -> int | None:
        """
        Return the class currently used for new annotations.
        """
        return self._active_class_id

    def refresh_classes(self) -> None:
        """
        Rebuild the class list from the active annotation session.

        The current drawing class is preserved when it still exists.
        Otherwise, the first available class becomes active.
        """
        session = self._controller.session

        previous_class_id = self._active_class_id

        self._class_list.blockSignals(True)

        try:
            self._class_list.clear()
            self._active_class_id = None

            if session is None:
                self._active_class_label.setText(
                    "Drawing class: None"
                )

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

                item.setToolTip(
                    f"Class ID: {class_definition.class_id}"
                )

                self._class_list.addItem(item)

            if self._class_list.count() == 0:
                self._active_class_label.setText(
                    "Drawing class: None"
                )

                self._clear_selected_annotation()
                return

            item_to_select = self._find_class_item(
                previous_class_id
            )

            if item_to_select is None:
                item_to_select = self._class_list.item(0)

            self._class_list.setCurrentItem(
                item_to_select
            )

        finally:
            self._class_list.blockSignals(False)

        current_item = self._class_list.currentItem()

        if current_item is not None:
            self._activate_item(current_item)

        self._refresh_selected_annotation_display()

    def set_active_class_id(
        self,
        class_id: int,
    ) -> None:
        """
        Select a class programmatically.

        This also updates the drawing class used by ImageCanvas.
        """
        session = self._controller.session

        if session is None:
            raise RuntimeError(
                "No annotation session is currently open."
            )

        if session.get_class(class_id) is None:
            raise ValueError(
                f"Class ID {class_id} is not defined "
                "in the current session."
            )

        class_item = self._find_class_item(class_id)

        if class_item is None:
            raise ValueError(
                f"Class ID {class_id} is not displayed "
                "in the class panel."
            )

        self._class_list.setCurrentItem(class_item)

    def clear(self) -> None:
        """
        Clear all class and annotation-selection information.
        """
        self._class_list.clear()

        self._active_class_id = None
        self._selected_annotation_index = None

        self._active_class_label.setText(
            "Drawing class: None"
        )

        self._selected_box_label.setText(
            "Selected box: None"
        )

        self._update_apply_button_state()

    def _configure_widgets(self) -> None:
        """
        Configure the panel's widgets.
        """
        self._title_label.setObjectName(
            "classPanelTitle"
        )

        self._class_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        self._class_list.setAlternatingRowColors(True)

        self._apply_class_button.setEnabled(False)

        self._apply_class_button.setToolTip(
            "Change the selected bounding box to the "
            "currently active drawing class."
        )

    def _build_layout(self) -> None:
        """
        Arrange the panel widgets vertically.
        """
        layout = QVBoxLayout(self)

        layout.addWidget(self._title_label)
        layout.addWidget(self._class_list, stretch=1)
        layout.addWidget(self._active_class_label)
        layout.addWidget(self._selected_box_label)
        layout.addWidget(self._apply_class_button)

        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """
        Connect class-panel and canvas events.
        """
        self._class_list.currentItemChanged.connect(
            self._handle_current_item_changed
        )

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

    def _handle_current_item_changed(
        self,
        current_item: QListWidgetItem | None,
        previous_item: QListWidgetItem | None,
    ) -> None:
        """
        Activate the class selected in the list.
        """
        del previous_item

        if current_item is None:
            self._active_class_id = None

            self._active_class_label.setText(
                "Drawing class: None"
            )

            self._update_apply_button_state()
            return

        self._activate_item(current_item)

    def _activate_item(
        self,
        item: QListWidgetItem,
    ) -> None:
        """
        Make one class-list item the active drawing class.
        """
        class_id_value = item.data(
            Qt.ItemDataRole.UserRole
        )

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
                f"Class ID {class_id} is not defined "
                "in the current session."
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
            f"{class_definition.class_id} - "
            f"{class_definition.name}"
        )

        self._update_apply_button_state()

        self.active_class_changed.emit(class_id)

    def _handle_annotation_selected(
        self,
        annotation_index: int,
    ) -> None:
        """
        Store and display information about the selected box.
        """
        self._selected_annotation_index = (
            annotation_index
        )

        self._refresh_selected_annotation_display()
        self._update_apply_button_state()

    def _clear_selected_annotation(self) -> None:
        """
        Clear the current bounding-box selection.
        """
        self._selected_annotation_index = None

        self._selected_box_label.setText(
            "Selected box: None"
        )

        self._update_apply_button_state()

    def _handle_annotation_deleted(
        self,
        deleted_index: int,
    ) -> None:
        """
        Refresh selected-box information after deletion.
        """
        del deleted_index

        current_selection = (
            self._canvas.selected_annotation_index
        )

        if current_selection is None:
            self._clear_selected_annotation()
            return

        self._handle_annotation_selected(
            current_selection
        )

    def _handle_annotation_updated(
        self,
        annotation_index: int,
    ) -> None:
        """
        Refresh the class display after an annotation changes.
        """
        if (
            self._selected_annotation_index
            == annotation_index
        ):
            self._refresh_selected_annotation_display()

    def _apply_active_class_to_selected_box(
        self,
    ) -> None:
        """
        Apply the active drawing class to the selected annotation.
        """
        annotation_index = (
            self._selected_annotation_index
        )

        class_id = self._active_class_id

        if annotation_index is None or class_id is None:
            return

        try:
            changed = (
                self._canvas
                .change_selected_annotation_class(
                    class_id
                )
            )

        except Exception as error:
            self.error_occurred.emit(str(error))
            return

        if not changed:
            return

        self._refresh_selected_annotation_display()

        self.annotation_class_changed.emit(
            annotation_index,
            class_id,
        )

    def _refresh_selected_annotation_display(
        self,
    ) -> None:
        """
        Display the selected annotation's current class.
        """
        annotation_index = (
            self._selected_annotation_index
        )

        image_record = self._controller.current_image
        session = self._controller.session

        if (
            annotation_index is None
            or image_record is None
            or session is None
        ):
            self._selected_box_label.setText(
                "Selected box: None"
            )

            self._update_apply_button_state()
            return

        if (
            annotation_index < 0
            or annotation_index
            >= len(image_record.annotations)
        ):
            self._clear_selected_annotation()
            return

        annotation = image_record.annotations[
            annotation_index
        ]

        class_name = session.get_class_name(
            annotation.class_id
        )

        if class_name is None:
            class_name = "Unknown"

        self._selected_box_label.setText(
            f"Selected box {annotation_index + 1}: "
            f"{annotation.class_id} - {class_name}"
        )

        self._update_apply_button_state()

    def _update_apply_button_state(self) -> None:
        """
        Enable class application only when it can succeed.
        """
        self._apply_class_button.setEnabled(
            self._active_class_id is not None
            and self._selected_annotation_index is not None
        )

    def _find_class_item(
        self,
        class_id: int | None,
    ) -> QListWidgetItem | None:
        """
        Find the list item associated with a class ID.
        """
        if class_id is None:
            return None

        for item_index in range(
            self._class_list.count()
        ):
            item = self._class_list.item(item_index)

            item_class_id = item.data(
                Qt.ItemDataRole.UserRole
            )

            if (
                item_class_id is not None
                and int(item_class_id) == class_id
            ):
                return item

        return None

    @staticmethod
    def _format_class_text(
        class_id: int,
        class_name: str,
    ) -> str:
        """
        Format a class for display in the list.
        """
        return f"{class_id}: {class_name}"