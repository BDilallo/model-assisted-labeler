from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from model_assisted_labeler.controllers.annotation_controller import (
    AnnotationController,
)
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.ui.bounding_box_item import BoundingBoxItem


class ImageCanvas(QGraphicsView):
    """
    Displays the current image and its editable bounding boxes.

    The canvas is responsible for translating GUI interaction into
    controller operations. It does not directly save annotations or
    modify ImageRecord objects.
    """

    annotation_selected = Signal(int)
    selection_cleared = Signal()
    annotation_created = Signal(int)
    annotation_updated = Signal(int)
    annotation_deleted = Signal(int)
    error_occurred = Signal(str)

    MINIMUM_BOX_SIZE = BoundingBoxItem.MINIMUM_BOX_SIZE
    ZOOM_FACTOR = 1.15
    MINIMUM_ZOOM = 0.05
    MAXIMUM_ZOOM = 40.0

    MODEL_PREDICTION_COLOR = QColor(45, 130, 255)
    EDITED_MODEL_COLOR = QColor(235, 70, 70)

    def __init__(
        self,
        controller: AnnotationController,
    ) -> None:
        super().__init__()

        self._controller = controller

        self._graphics_scene = QGraphicsScene(self)
        self._image_item: QGraphicsPixmapItem | None = None
        self._annotation_items: dict[int, BoundingBoxItem] = {}

        self._active_class_id: int | None = None
        self._selected_annotation_index: int | None = None

        self._is_drawing = False
        self._drawing_start: QPointF | None = None
        self._drawing_preview: QGraphicsRectItem | None = None

        self._suppress_selection_signals = False
        self._fit_after_resize = False

        self._configure_view()
        self._connect_signals()

    @property
    def active_class_id(self) -> int | None:
        """Return the class used when drawing a new box."""
        return self._active_class_id

    @property
    def selected_annotation_index(self) -> int | None:
        """Return the currently selected annotation index."""
        return self._selected_annotation_index

    @property
    def has_image(self) -> bool:
        """Return True when an image is displayed."""
        return self._image_item is not None

    def set_active_class_id(
        self,
        class_id: int,
    ) -> None:
        """
        Set the class assigned to newly drawn bounding boxes.
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

        self._active_class_id = class_id

    def display_current_image(self) -> None:
        """
        Display the controller's current image and annotations.

        This method should be called after opening a session or moving
        to a different image.
        """
        image_record = self._controller.current_image

        self.clear()

        if image_record is None:
            return

        pixmap = QPixmap(str(image_record.image_path))

        if pixmap.isNull():
            self.error_occurred.emit(
                f"Could not display image: {image_record.image_path}"
            )
            return

        self._graphics_scene.setSceneRect(
            0.0,
            0.0,
            float(image_record.width),
            float(image_record.height),
        )

        self._image_item = self._graphics_scene.addPixmap(pixmap)
        self._image_item.setZValue(-1000.0)

        self.refresh_annotations()
        self.fit_to_image()

    def refresh_annotations(self) -> None:
        """
        Rebuild displayed boxes from the current ImageRecord.

        The image itself and the current zoom level are preserved.
        """
        image_record = self._controller.current_image
        session = self._controller.session

        self._remove_annotation_items()

        if image_record is None or session is None:
            return

        for annotation_index, bounding_box in enumerate(
            image_record.annotations
        ):
            class_name = session.get_class_name(
                bounding_box.class_id
            )

            if class_name is None:
                class_name = "Unknown"

            self._add_annotation_item(
                annotation_index=annotation_index,
                bounding_box=bounding_box,
                class_name=class_name,
            )

    def clear(self) -> None:
        """Remove the displayed image without using scene.clear().

        BoundingBoxItem owns child resize-handle graphics items. Calling
        QGraphicsScene.clear() destroys that complete C++ hierarchy in one
        operation while Python dictionaries can still hold wrappers for the
        deleted items. Removing each top-level item explicitly avoids that
        ownership race during navigation.
        """
        self._cancel_drawing()
        self._suppress_selection_signals = True
        signal_blocker = QSignalBlocker(self._graphics_scene)

        try:
            self._graphics_scene.clearSelection()
            self._detach_annotation_items()

            image_item = self._image_item
            self._image_item = None

            if (
                image_item is not None
                and image_item.scene() is self._graphics_scene
            ):
                self._graphics_scene.removeItem(image_item)

            self._graphics_scene.setSceneRect(QRectF())

        finally:
            del signal_blocker
            self._suppress_selection_signals = False

        self._selected_annotation_index = None
        self.resetTransform()
        self._fit_after_resize = False
        self.selection_cleared.emit()

    def fit_to_image(self) -> None:
        """Fit the entire displayed image inside the viewport."""
        if self._image_item is None:
            return

        image_bounds = self._graphics_scene.sceneRect()

        if image_bounds.isEmpty():
            return

        self.resetTransform()
        self.fitInView(
            image_bounds,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        self._fit_after_resize = True

    def delete_selected_annotation(self) -> bool:
        """
        Delete the currently selected annotation.

        Returns True when an annotation was deleted.
        """
        annotation_index = self._selected_annotation_index

        if annotation_index is None:
            return False

        try:
            self._controller.remove_annotation(annotation_index)

        except Exception as error:
            self.error_occurred.emit(str(error))
            return False

        self.refresh_annotations()
        self.annotation_deleted.emit(annotation_index)

        return True

    def change_selected_annotation_class(
        self,
        class_id: int,
    ) -> bool:
        """
        Apply a class ID to the selected annotation.

        Returns True when a selected annotation was changed.
        """
        annotation_index = self._selected_annotation_index

        if annotation_index is None:
            return False

        try:
            self._controller.change_annotation_class(
                index=annotation_index,
                class_id=class_id,
            )

            session = self._controller.session

            if session is None:
                raise RuntimeError(
                    "No annotation session is currently open."
                )

            class_name = session.get_class_name(class_id)

            if class_name is None:
                raise ValueError(
                    f"Class ID {class_id} is not defined "
                    "in the current session."
                )

            image_record = self._controller.current_image

            if image_record is None:
                raise RuntimeError(
                    "No image is currently selected."
                )

            stored_box = image_record.annotations[
                annotation_index
            ]

            annotation_item = self._annotation_items.get(
                annotation_index
            )

            if annotation_item is not None:
                annotation_item.set_class(
                    class_id=class_id,
                    class_name=class_name,
                )
                annotation_item.set_color(
                    self._color_for_annotation(stored_box)
                )

        except Exception as error:
            self.error_occurred.emit(str(error))
            return False

        self.annotation_updated.emit(annotation_index)

        return True

    def mousePressEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        """
        Start drawing on empty image space or pass interaction to an
        existing graphics item.
        """
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        if self._controller.current_image is None:
            super().mousePressEvent(event)
            return

        view_position = event.position().toPoint()
        clicked_item = self.itemAt(view_position)

        if self._find_bounding_box_item(clicked_item) is not None:
            super().mousePressEvent(event)
            return

        scene_position = self.mapToScene(view_position)

        if not self._point_is_inside_image(scene_position):
            super().mousePressEvent(event)
            return

        if self._active_class_id is None:
            self.error_occurred.emit(
                "Select an annotation class before drawing a box."
            )
            return

        self._graphics_scene.clearSelection()

        self._is_drawing = True
        self._drawing_start = scene_position

        preview_pen = QPen(QColor(255, 255, 255))
        preview_pen.setWidthF(2.0)
        preview_pen.setStyle(Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)

        self._drawing_preview = QGraphicsRectItem()
        self._drawing_preview.setPen(preview_pen)
        self._drawing_preview.setBrush(
            Qt.BrushStyle.NoBrush
        )
        self._drawing_preview.setZValue(2000.0)

        self._graphics_scene.addItem(
            self._drawing_preview
        )

        event.accept()

    def mouseMoveEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        """Update the temporary rectangle while drawing."""
        if not self._is_drawing:
            super().mouseMoveEvent(event)
            return

        if (
            self._drawing_start is None
            or self._drawing_preview is None
        ):
            self._cancel_drawing()
            return

        scene_position = self.mapToScene(
            event.position().toPoint()
        )

        scene_position = self._clamp_point_to_image(
            scene_position
        )

        preview_rectangle = QRectF(
            self._drawing_start,
            scene_position,
        ).normalized()

        self._drawing_preview.setRect(
            preview_rectangle
        )

        event.accept()

    def mouseReleaseEvent(
        self,
        event: QMouseEvent,
    ) -> None:
        """Finish drawing and create a new annotation."""
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not self._is_drawing
        ):
            super().mouseReleaseEvent(event)
            return

        if (
            self._drawing_start is None
            or self._drawing_preview is None
        ):
            self._cancel_drawing()
            return

        scene_position = self.mapToScene(
            event.position().toPoint()
        )

        scene_position = self._clamp_point_to_image(
            scene_position
        )

        final_rectangle = QRectF(
            self._drawing_start,
            scene_position,
        ).normalized()

        active_class_id = self._active_class_id

        self._cancel_drawing()

        if (
            final_rectangle.width() < self.MINIMUM_BOX_SIZE
            or final_rectangle.height() < self.MINIMUM_BOX_SIZE
        ):
            event.accept()
            return

        if active_class_id is None:
            event.accept()
            return

        bounding_box = BoundingBox(
            class_id=active_class_id,
            x1=final_rectangle.left(),
            y1=final_rectangle.top(),
            x2=final_rectangle.right(),
            y2=final_rectangle.bottom(),
            confidence=None,
            source="manual",
        )

        try:
            self._controller.add_annotation(bounding_box)

            image_record = self._controller.current_image
            session = self._controller.session

            if image_record is None or session is None:
                raise RuntimeError(
                    "The annotation session is no longer available."
                )

            annotation_index = len(
                image_record.annotations
            ) - 1

            stored_box = image_record.annotations[
                annotation_index
            ]

            class_name = session.get_class_name(
                stored_box.class_id
            )

            if class_name is None:
                class_name = "Unknown"

            annotation_item = self._add_annotation_item(
                annotation_index=annotation_index,
                bounding_box=stored_box,
                class_name=class_name,
            )

            annotation_item.setSelected(True)
            annotation_item.setFocus()

        except Exception as error:
            self.error_occurred.emit(str(error))
            event.accept()
            return

        self.annotation_created.emit(annotation_index)
        event.accept()

    def wheelEvent(
        self,
        event: QWheelEvent,
    ) -> None:
        """Zoom toward the mouse pointer using the wheel."""
        if self._image_item is None:
            super().wheelEvent(event)
            return

        current_zoom = self.transform().m11()

        if event.angleDelta().y() > 0:
            requested_factor = self.ZOOM_FACTOR
        else:
            requested_factor = 1.0 / self.ZOOM_FACTOR

        proposed_zoom = current_zoom * requested_factor

        if proposed_zoom < self.MINIMUM_ZOOM:
            requested_factor = (
                self.MINIMUM_ZOOM / current_zoom
            )

        elif proposed_zoom > self.MAXIMUM_ZOOM:
            requested_factor = self.MAXIMUM_ZOOM / current_zoom

        self.scale(
            requested_factor,
            requested_factor,
        )

        self._fit_after_resize = False
        event.accept()

    def keyPressEvent(
        self,
        event: QKeyEvent,
    ) -> None:
        """Handle deletion, cancellation, and fit-to-view shortcuts."""
        if event.key() in {
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        }:
            if self.delete_selected_annotation():
                event.accept()
                return

        elif event.key() == Qt.Key.Key_Escape:
            if self._is_drawing:
                self._cancel_drawing()
            else:
                self._graphics_scene.clearSelection()

            event.accept()
            return

        elif event.key() == Qt.Key.Key_F:
            self.fit_to_image()
            event.accept()
            return

        super().keyPressEvent(event)

    def resizeEvent(
        self,
        event: QResizeEvent,
    ) -> None:
        """Keep an initially fitted image fitted as the view resizes."""
        super().resizeEvent(event)

        if self._fit_after_resize:
            self.fit_to_image()

    def _configure_view(self) -> None:
        """Configure graphics-view behavior and appearance."""
        self.setScene(self._graphics_scene)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )

        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
        )

        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        self.setBackgroundBrush(
            QColor(35, 35, 35)
        )

        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor
        )

    def _connect_signals(self) -> None:
        """Connect scene-selection changes to canvas signals."""
        self._graphics_scene.selectionChanged.connect(
            self._handle_scene_selection_changed
        )

    def _add_annotation_item(
        self,
        annotation_index: int,
        bounding_box: BoundingBox,
        class_name: str,
    ) -> BoundingBoxItem:
        """Create and register one editable graphics item."""
        image_record = self._controller.current_image

        if image_record is None:
            raise RuntimeError(
                "No image is currently displayed."
            )

        item = BoundingBoxItem(
            annotation_index=annotation_index,
            bounding_box=bounding_box,
            class_name=class_name,
            image_width=image_record.width,
            image_height=image_record.height,
            geometry_changed_callback=(
                self._handle_item_geometry_changed
            ),
            color=self._color_for_annotation(
                bounding_box
            ),
        )

        item.setZValue(1000.0 + annotation_index)

        self._graphics_scene.addItem(item)
        self._annotation_items[annotation_index] = item

        return item

    def _remove_annotation_items(self) -> None:
        """Remove all box items while keeping the image item."""
        self._suppress_selection_signals = True
        signal_blocker = QSignalBlocker(self._graphics_scene)

        try:
            self._graphics_scene.clearSelection()
            self._detach_annotation_items()

        finally:
            del signal_blocker
            self._suppress_selection_signals = False

        self._selected_annotation_index = None
        self.selection_cleared.emit()

    def _detach_annotation_items(self) -> None:
        """Make box hierarchies inert and remove them one at a time."""
        annotation_items = list(self._annotation_items.values())
        self._annotation_items.clear()

        for item in annotation_items:
            item.prepare_for_removal()

            if item.scene() is self._graphics_scene:
                self._graphics_scene.removeItem(item)

    def _handle_scene_selection_changed(self) -> None:
        """Emit the annotation index selected in the scene."""
        if self._suppress_selection_signals:
            return

        selected_items = [
            item
            for item in self._graphics_scene.selectedItems()
            if isinstance(item, BoundingBoxItem)
        ]

        if not selected_items:
            if self._selected_annotation_index is not None:
                self._selected_annotation_index = None
                self.selection_cleared.emit()

            return

        selected_item = selected_items[0]

        if len(selected_items) > 1:
            self._suppress_selection_signals = True

            try:
                for extra_item in selected_items[1:]:
                    extra_item.setSelected(False)

            finally:
                self._suppress_selection_signals = False

        self._selected_annotation_index = (
            selected_item.annotation_index
        )

        self.annotation_selected.emit(
            selected_item.annotation_index
        )

    def _handle_item_geometry_changed(
        self,
        annotation_index: int,
        updated_box: BoundingBox,
    ) -> None:
        """Pass a moved or resized box to the controller."""
        try:
            self._controller.update_annotation(
                index=annotation_index,
                updated_box=updated_box,
            )

            image_record = self._controller.current_image

            if image_record is None:
                raise RuntimeError(
                    "No image is currently selected."
                )

            stored_box = image_record.annotations[
                annotation_index
            ]

            annotation_item = self._annotation_items.get(
                annotation_index
            )

            if annotation_item is not None:
                annotation_item.update_from_bounding_box(
                    stored_box
                )
                annotation_item.set_color(
                    self._color_for_annotation(stored_box)
                )

        except Exception as error:
            self.error_occurred.emit(str(error))
            self.refresh_annotations()
            return

        self.annotation_updated.emit(annotation_index)

    @staticmethod
    def _find_bounding_box_item(
        graphics_item: QGraphicsItem | None,
    ) -> BoundingBoxItem | None:
        """
        Return a bounding box when an item or one of its children
        was clicked.

        Resize controls are child graphics items, so checking parent
        items prevents the canvas from treating a handle click as the
        start of a brand-new annotation.
        """
        current_item = graphics_item

        while current_item is not None:
            if isinstance(current_item, BoundingBoxItem):
                return current_item

            current_item = current_item.parentItem()

        return None

    def _point_is_inside_image(
        self,
        point: QPointF,
    ) -> bool:
        """Return True when a scene point is inside the image."""
        if self._image_item is None:
            return False

        return self._graphics_scene.sceneRect().contains(point)

    def _clamp_point_to_image(
        self,
        point: QPointF,
    ) -> QPointF:
        """Keep a scene point within the displayed image bounds."""
        bounds = self._graphics_scene.sceneRect()

        return QPointF(
            min(
                max(point.x(), bounds.left()),
                bounds.right(),
            ),
            min(
                max(point.y(), bounds.top()),
                bounds.bottom(),
            ),
        )

    def _cancel_drawing(self) -> None:
        """Remove the drawing preview and reset drawing state."""
        if self._drawing_preview is not None:
            if self._drawing_preview.scene() is not None:
                self._graphics_scene.removeItem(
                    self._drawing_preview
                )

        self._drawing_preview = None
        self._drawing_start = None
        self._is_drawing = False

    @classmethod
    def _color_for_annotation(
        cls,
        bounding_box: BoundingBox,
    ) -> QColor:
        """Return a color based on annotation origin and edit state."""
        if (
            bounding_box.source
            == AnnotationController.MODEL_SOURCE
        ):
            return QColor(cls.MODEL_PREDICTION_COLOR)

        if (
            bounding_box.source
            == AnnotationController.MODEL_EDITED_SOURCE
        ):
            return QColor(cls.EDITED_MODEL_COLOR)

        return cls._color_for_class(
            bounding_box.class_id
        )

    @staticmethod
    def _color_for_class(
        class_id: int,
    ) -> QColor:
        """Return a stable visible color for a class ID."""
        hue = (class_id * 47) % 360

        return QColor.fromHsv(
            hue,
            210,
            255,
        )
