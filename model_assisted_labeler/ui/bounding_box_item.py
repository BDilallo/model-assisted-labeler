from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from model_assisted_labeler.geometry.bounding_box_geometry import (
    BoundingBoxGeometry,
    ResizeHandle,
)
from model_assisted_labeler.models.bounding_box import BoundingBox
from model_assisted_labeler.ui.resize_handle_item import ResizeHandleItem


class BoundingBoxItem(QGraphicsRectItem):
    """
    Displays and coordinates one editable annotation rectangle.

    Rendering and mouse-event coordination remain in this UI class.
    Movement and resize calculations are delegated to the geometry layer.
    """

    MINIMUM_BOX_SIZE = 4.0

    def __init__(
        self,
        annotation_index: int,
        bounding_box: BoundingBox,
        class_name: str,
        image_width: int,
        image_height: int,
        geometry_changed_callback: (
            Callable[[int, BoundingBox], None] | None
        ) = None,
        color: QColor | None = None,
    ) -> None:
        super().__init__()

        if annotation_index < 0:
            raise ValueError(
                "Annotation index cannot be negative."
            )

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Image dimensions must be greater than zero."
            )

        if not bounding_box.is_valid():
            raise ValueError(
                "BoundingBoxItem requires a valid BoundingBox."
            )

        cleaned_class_name = class_name.strip()

        if not cleaned_class_name:
            raise ValueError("Class name cannot be empty.")

        self._annotation_index = annotation_index
        self._bounding_box = bounding_box
        self._class_name = cleaned_class_name
        self._geometry_changed_callback = geometry_changed_callback

        self._image_bounds = BoundingBoxGeometry(
            left=0.0,
            top=0.0,
            right=float(image_width),
            bottom=float(image_height),
        )

        self._geometry_before_interaction: (
            BoundingBoxGeometry | None
        ) = None

        self._resize_start_geometry: (
            BoundingBoxGeometry | None
        ) = None

        self._active_resize_handle: ResizeHandle | None = None
        self._applying_geometry = False

        if color is None:
            color = QColor(0, 220, 100)

        box_pen = QPen(color)
        box_pen.setWidthF(3.0)
        box_pen.setCosmetic(True)

        self.setPen(box_pen)
        self.setBrush(Qt.BrushStyle.NoBrush)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        self._resize_handles = {
            handle: ResizeHandleItem(
                handle=handle,
                color=color,
                parent=self,
            )
            for handle in ResizeHandle
        }

        self.update_from_bounding_box(bounding_box)

    @property
    def annotation_index(self) -> int:
        """Return the annotation's position in the image record."""
        return self._annotation_index

    @property
    def class_name(self) -> str:
        """Return the displayed annotation class name."""
        return self._class_name

    def set_annotation_index(
        self,
        annotation_index: int,
    ) -> None:
        """Update the annotation index associated with the item."""
        if annotation_index < 0:
            raise ValueError(
                "Annotation index cannot be negative."
            )

        self._annotation_index = annotation_index

    def set_color(
        self,
        color: QColor,
    ) -> None:
        """Update the rectangle and resize-handle colors."""
        box_pen = QPen(color)
        box_pen.setWidthF(3.0)
        box_pen.setCosmetic(True)
        self.setPen(box_pen)

        for handle_item in self._resize_handles.values():
            handle_item.set_color(color)

        self.update()

    def set_class(
        self,
        class_id: int,
        class_name: str,
    ) -> None:
        """Change the item's class ID and displayed class name."""
        if class_id < 0:
            raise ValueError("Class ID cannot be negative.")

        cleaned_name = class_name.strip()

        if not cleaned_name:
            raise ValueError("Class name cannot be empty.")

        self._bounding_box.class_id = class_id
        self._class_name = cleaned_name
        self.update()

    def update_from_bounding_box(
        self,
        bounding_box: BoundingBox,
    ) -> None:
        """Replace the displayed geometry using annotation data."""
        bounding_box.normalize_coordinates()
        bounding_box.clamp(
            image_width=int(self._image_bounds.width),
            image_height=int(self._image_bounds.height),
        )

        if not bounding_box.is_valid():
            raise ValueError(
                "Cannot display an invalid bounding box."
            )

        geometry = BoundingBoxGeometry(
            left=bounding_box.x1,
            top=bounding_box.y1,
            right=bounding_box.x2,
            bottom=bounding_box.y2,
        ).clamp_to_bounds(self._image_bounds)

        if not geometry.is_valid():
            raise ValueError(
                "Cannot display collapsed bounding-box geometry."
            )

        self._bounding_box = bounding_box
        self._apply_geometry(geometry)

    def to_bounding_box(self) -> BoundingBox:
        """Convert the current scene geometry into annotation data."""
        geometry = self._current_geometry()

        return BoundingBox(
            class_id=self._bounding_box.class_id,
            x1=geometry.left,
            y1=geometry.top,
            x2=geometry.right,
            y2=geometry.bottom,
            confidence=self._bounding_box.confidence,
            source=self._bounding_box.source,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Draw the rectangle and its class label."""
        del option, widget

        painter.save()
        painter.setPen(self.pen())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect())

        painter.drawText(
            self.rect().adjusted(
                6.0,
                4.0,
                -6.0,
                -4.0,
            ),
            (
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignTop
            ),
            self._class_name,
        )

        painter.restore()

    def mousePressEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """Begin moving the box when its main rectangle is dragged."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSelected(True)
            self.setFocus()
            self._geometry_before_interaction = (
                self._current_geometry()
            )

        super().mousePressEvent(event)

    def mouseReleaseEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """Finish a move operation and report changed geometry."""
        super().mouseReleaseEvent(event)
        self._finish_geometry_interaction()

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        """Clamp movement and show handles when the box is selected."""
        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and not self._applying_geometry
            and isinstance(value, QPointF)
        ):
            current_geometry = self._current_geometry()
            moved_geometry = current_geometry.move_to_clamped(
                x=value.x(),
                y=value.y(),
                bounds=self._image_bounds,
            )

            return QPointF(
                moved_geometry.left,
                moved_geometry.top,
            )

        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
        ):
            self._set_handles_visible(bool(value))
            self.update()

        return super().itemChange(change, value)

    def begin_resize(
        self,
        handle: ResizeHandle,
    ) -> None:
        """Begin resizing from one corner control."""
        self.setSelected(True)
        self.setFocus()

        current_geometry = self._current_geometry()

        self._geometry_before_interaction = current_geometry
        self._resize_start_geometry = current_geometry
        self._active_resize_handle = handle

    def continue_resize(
        self,
        scene_position: QPointF,
    ) -> None:
        """Resize the box toward a scene position."""
        if (
            self._resize_start_geometry is None
            or self._active_resize_handle is None
        ):
            return

        resized_geometry = (
            self._resize_start_geometry.resize_from_handle(
                handle=self._active_resize_handle,
                pointer_x=scene_position.x(),
                pointer_y=scene_position.y(),
                bounds=self._image_bounds,
                minimum_size=self.MINIMUM_BOX_SIZE,
            )
        )

        self._apply_geometry(resized_geometry)

    def finish_resize(self) -> None:
        """Finish resizing and report changed geometry."""
        self._resize_start_geometry = None
        self._active_resize_handle = None
        self._finish_geometry_interaction()

    def _apply_geometry(
        self,
        geometry: BoundingBoxGeometry,
    ) -> None:
        """Apply image-coordinate geometry to the graphics item."""
        geometry = geometry.clamp_to_bounds(
            self._image_bounds
        )

        if not geometry.is_valid():
            raise ValueError(
                "Bounding-box geometry must have positive dimensions."
            )

        self._applying_geometry = True

        try:
            self.setPos(geometry.left, geometry.top)
            self.setRect(
                QRectF(
                    0.0,
                    0.0,
                    geometry.width,
                    geometry.height,
                )
            )

        finally:
            self._applying_geometry = False

        self._position_resize_handles()
        self.update()

    def _current_geometry(self) -> BoundingBoxGeometry:
        """Return the item's current rectangle in scene coordinates."""
        return BoundingBoxGeometry.from_position_and_size(
            x=self.pos().x(),
            y=self.pos().y(),
            width=self.rect().width(),
            height=self.rect().height(),
        )

    def _position_resize_handles(self) -> None:
        """Place each resize control at its matching local corner."""
        rectangle = self.rect()

        handle_positions = {
            ResizeHandle.TOP_LEFT: rectangle.topLeft(),
            ResizeHandle.TOP_RIGHT: rectangle.topRight(),
            ResizeHandle.BOTTOM_LEFT: rectangle.bottomLeft(),
            ResizeHandle.BOTTOM_RIGHT: rectangle.bottomRight(),
        }

        for handle, position in handle_positions.items():
            self._resize_handles[handle].setPos(position)

    def _set_handles_visible(
        self,
        visible: bool,
    ) -> None:
        """Show or hide all four corner controls."""
        for handle_item in self._resize_handles.values():
            handle_item.setVisible(visible)

    def prepare_for_removal(self) -> None:
        """Detach callbacks and interaction state before scene removal.

        QGraphicsScene can otherwise destroy the C++ item hierarchy while
        Python callbacks still reference this box and its child handles.
        Explicitly making the item inert is safer during image navigation,
        particularly on Windows.
        """
        self._geometry_changed_callback = None
        self._geometry_before_interaction = None
        self._resize_start_geometry = None
        self._active_resize_handle = None
        self._set_handles_visible(False)
        self.setSelected(False)
        self.clearFocus()
        self.setEnabled(False)

    def _finish_geometry_interaction(self) -> None:
        """Notify the canvas when a move or resize changed the box."""
        previous_geometry = self._geometry_before_interaction
        self._geometry_before_interaction = None

        if previous_geometry is None:
            return

        current_geometry = self._current_geometry()

        if current_geometry.is_close_to(previous_geometry):
            return

        if self._geometry_changed_callback is None:
            return

        self._geometry_changed_callback(
            self._annotation_index,
            self.to_bounding_box(),
        )
