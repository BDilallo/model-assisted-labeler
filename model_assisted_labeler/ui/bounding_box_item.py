from collections.abc import Callable
from enum import Enum, auto

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from model_assisted_labeler.models.bounding_box import BoundingBox


class ResizeHandle(Enum):
    """
    Identifies which corner of a bounding box is being resized.
    """

    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


class BoundingBoxItem(QGraphicsRectItem):
    """
    Displays one editable bounding box inside a QGraphicsScene.

    The item supports:

    - Selection
    - Dragging
    - Corner resizing
    - Image-boundary enforcement
    - Conversion back into a BoundingBox
    - Notification when its geometry changes
    """

    HANDLE_SIZE = 10.0
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

        self._annotation_index = annotation_index
        self._bounding_box = bounding_box
        self._class_name = class_name

        self._image_bounds = QRectF(
            0.0,
            0.0,
            float(image_width),
            float(image_height),
        )

        self._geometry_changed_callback = (
            geometry_changed_callback
        )

        self._active_handle: ResizeHandle | None = None
        self._interaction_start_rect: QRectF | None = None
        self._geometry_before_interaction: QRectF | None = None

        self._applying_geometry = False

        if color is None:
            color = QColor(0, 220, 100)

        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)

        self.setPen(pen)
        self.setBrush(Qt.BrushStyle.NoBrush)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        self.setAcceptHoverEvents(True)

        self.update_from_bounding_box(bounding_box)

    @property
    def annotation_index(self) -> int:
        """
        Return the annotation's position in the ImageRecord list.
        """
        return self._annotation_index

    @property
    def class_name(self) -> str:
        """Return the displayed annotation class name."""
        return self._class_name

    def set_annotation_index(
        self,
        annotation_index: int,
    ) -> None:
        """
        Update the annotation index associated with the item.
        """
        if annotation_index < 0:
            raise ValueError(
                "Annotation index cannot be negative."
            )

        self._annotation_index = annotation_index

    def set_class(
        self,
        class_id: int,
        class_name: str,
    ) -> None:
        """
        Change the item's class ID and displayed class name.
        """
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
        """
        Replace the displayed geometry using a BoundingBox.

        This is used when the canvas is refreshed from application
        state rather than from a direct mouse interaction.
        """
        bounding_box.normalize_coordinates()

        bounding_box.clamp(
            image_width=int(self._image_bounds.width()),
            image_height=int(self._image_bounds.height()),
        )

        if not bounding_box.is_valid():
            raise ValueError(
                "Cannot display an invalid bounding box."
            )

        self._bounding_box = bounding_box

        scene_rectangle = QRectF(
            bounding_box.x1,
            bounding_box.y1,
            bounding_box.width,
            bounding_box.height,
        )

        self._apply_scene_rectangle(scene_rectangle)

    def to_bounding_box(self) -> BoundingBox:
        """
        Convert the item's current scene geometry into annotation data.
        """
        scene_rectangle = self._current_scene_rectangle()

        return BoundingBox(
            class_id=self._bounding_box.class_id,
            x1=scene_rectangle.left(),
            y1=scene_rectangle.top(),
            x2=scene_rectangle.right(),
            y2=scene_rectangle.bottom(),
            confidence=self._bounding_box.confidence,
            source=self._bounding_box.source,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """
        Draw the rectangle, class name, and resize handles.
        """
        super().paint(painter, option, widget)

        painter.save()

        painter.setPen(self.pen())
        painter.drawText(
            self.rect().adjusted(
                4.0,
                2.0,
                -4.0,
                -2.0,
            ),
            (
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignTop
            ),
            self._class_name,
        )

        if self.isSelected():
            painter.setBrush(self.pen().color())
            painter.setPen(Qt.PenStyle.NoPen)

            for handle_rectangle in self._handle_rectangles().values():
                painter.drawRect(handle_rectangle)

        painter.restore()

    def hoverMoveEvent(
        self,
        event: QGraphicsSceneHoverEvent,
    ) -> None:
        """
        Change the cursor when hovering over a resize handle.
        """
        handle = self._handle_at(event.pos())

        if handle in {
            ResizeHandle.TOP_LEFT,
            ResizeHandle.BOTTOM_RIGHT,
        }:
            self.setCursor(
                QCursor(
                    Qt.CursorShape.SizeFDiagCursor
                )
            )

        elif handle in {
            ResizeHandle.TOP_RIGHT,
            ResizeHandle.BOTTOM_LEFT,
        }:
            self.setCursor(
                QCursor(
                    Qt.CursorShape.SizeBDiagCursor
                )
            )

        else:
            self.setCursor(
                QCursor(Qt.CursorShape.SizeAllCursor)
            )

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(
        self,
        event: QGraphicsSceneHoverEvent,
    ) -> None:
        """Restore the normal cursor when leaving the item."""
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """
        Begin either a move or corner-resize operation.
        """
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        self.setSelected(True)
        self.setFocus()

        self._geometry_before_interaction = (
            self._current_scene_rectangle()
        )

        selected_handle = self._handle_at(event.pos())

        if selected_handle is not None:
            self._active_handle = selected_handle
            self._interaction_start_rect = (
                self._current_scene_rectangle()
            )

            event.accept()
            return

        self._active_handle = None
        self._interaction_start_rect = None

        super().mousePressEvent(event)

    def mouseMoveEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """
        Resize from the active corner or allow Qt to move the item.
        """
        if (
            self._active_handle is not None
            and self._interaction_start_rect is not None
        ):
            self._resize_from_scene_position(
                event.scenePos()
            )

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """
        Finish the interaction and report changed geometry.
        """
        was_resizing = self._active_handle is not None

        if not was_resizing:
            super().mouseReleaseEvent(event)
        else:
            event.accept()

        self._active_handle = None
        self._interaction_start_rect = None

        current_geometry = self._current_scene_rectangle()

        if (
            self._geometry_before_interaction is not None
            and current_geometry
            != self._geometry_before_interaction
        ):
            self._notify_geometry_changed()

        self._geometry_before_interaction = None

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        """
        Restrict movement so the box remains inside the image.
        """
        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and not self._applying_geometry
        ):
            proposed_position = value

            if isinstance(proposed_position, QPointF):
                return self._clamp_item_position(
                    proposed_position
                )

        return super().itemChange(change, value)

    def _handle_rectangles(
        self,
    ) -> dict[ResizeHandle, QRectF]:
        """
        Return local rectangles for the four resize handles.
        """
        rectangle = self.rect()
        handle_size = self.HANDLE_SIZE
        half_handle = handle_size / 2.0

        return {
            ResizeHandle.TOP_LEFT: QRectF(
                rectangle.left() - half_handle,
                rectangle.top() - half_handle,
                handle_size,
                handle_size,
            ),
            ResizeHandle.TOP_RIGHT: QRectF(
                rectangle.right() - half_handle,
                rectangle.top() - half_handle,
                handle_size,
                handle_size,
            ),
            ResizeHandle.BOTTOM_LEFT: QRectF(
                rectangle.left() - half_handle,
                rectangle.bottom() - half_handle,
                handle_size,
                handle_size,
            ),
            ResizeHandle.BOTTOM_RIGHT: QRectF(
                rectangle.right() - half_handle,
                rectangle.bottom() - half_handle,
                handle_size,
                handle_size,
            ),
        }

    def _handle_at(
        self,
        local_position: QPointF,
    ) -> ResizeHandle | None:
        """
        Return the resize handle under a local item position.
        """
        if not self.isSelected():
            return None

        for handle, rectangle in (
            self._handle_rectangles().items()
        ):
            if rectangle.contains(local_position):
                return handle

        return None

    def _resize_from_scene_position(
        self,
        scene_position: QPointF,
    ) -> None:
        """
        Resize the active corner toward a scene position.
        """
        if (
            self._active_handle is None
            or self._interaction_start_rect is None
        ):
            return

        starting_rectangle = QRectF(
            self._interaction_start_rect
        )

        left = starting_rectangle.left()
        top = starting_rectangle.top()
        right = starting_rectangle.right()
        bottom = starting_rectangle.bottom()

        minimum_size = self.MINIMUM_BOX_SIZE
        bounds = self._image_bounds

        if self._active_handle == ResizeHandle.TOP_LEFT:
            left = max(
                bounds.left(),
                min(
                    scene_position.x(),
                    right - minimum_size,
                ),
            )

            top = max(
                bounds.top(),
                min(
                    scene_position.y(),
                    bottom - minimum_size,
                ),
            )

        elif self._active_handle == ResizeHandle.TOP_RIGHT:
            right = min(
                bounds.right(),
                max(
                    scene_position.x(),
                    left + minimum_size,
                ),
            )

            top = max(
                bounds.top(),
                min(
                    scene_position.y(),
                    bottom - minimum_size,
                ),
            )

        elif self._active_handle == ResizeHandle.BOTTOM_LEFT:
            left = max(
                bounds.left(),
                min(
                    scene_position.x(),
                    right - minimum_size,
                ),
            )

            bottom = min(
                bounds.bottom(),
                max(
                    scene_position.y(),
                    top + minimum_size,
                ),
            )

        elif self._active_handle == ResizeHandle.BOTTOM_RIGHT:
            right = min(
                bounds.right(),
                max(
                    scene_position.x(),
                    left + minimum_size,
                ),
            )

            bottom = min(
                bounds.bottom(),
                max(
                    scene_position.y(),
                    top + minimum_size,
                ),
            )

        resized_rectangle = QRectF(
            QPointF(left, top),
            QPointF(right, bottom),
        )

        self._apply_scene_rectangle(
            resized_rectangle.normalized()
        )

    def _apply_scene_rectangle(
        self,
        scene_rectangle: QRectF,
    ) -> None:
        """
        Set the item's position and local rectangle from scene geometry.
        """
        clamped_rectangle = (
            scene_rectangle.intersected(
                self._image_bounds
            )
        )

        if (
            clamped_rectangle.width()
            < self.MINIMUM_BOX_SIZE
            or clamped_rectangle.height()
            < self.MINIMUM_BOX_SIZE
        ):
            raise ValueError(
                "Bounding box is smaller than the minimum size."
            )

        self._applying_geometry = True

        try:
            self.setPos(clamped_rectangle.topLeft())

            self.setRect(
                0.0,
                0.0,
                clamped_rectangle.width(),
                clamped_rectangle.height(),
            )

        finally:
            self._applying_geometry = False

        self.update()

    def _current_scene_rectangle(self) -> QRectF:
        """
        Return annotation geometry in scene coordinates.
        """
        return QRectF(
            self.pos().x(),
            self.pos().y(),
            self.rect().width(),
            self.rect().height(),
        )

    def _clamp_item_position(
        self,
        proposed_position: QPointF,
    ) -> QPointF:
        """
        Keep a moved item completely inside the image.
        """
        maximum_x = (
            self._image_bounds.right()
            - self.rect().width()
        )

        maximum_y = (
            self._image_bounds.bottom()
            - self.rect().height()
        )

        clamped_x = min(
            max(
                proposed_position.x(),
                self._image_bounds.left(),
            ),
            maximum_x,
        )

        clamped_y = min(
            max(
                proposed_position.y(),
                self._image_bounds.top(),
            ),
            maximum_y,
        )

        return QPointF(clamped_x, clamped_y)

    def _notify_geometry_changed(self) -> None:
        """
        Send updated annotation geometry to the canvas/controller.
        """
        if self._geometry_changed_callback is None:
            return

        self._geometry_changed_callback(
            self._annotation_index,
            self.to_bounding_box(),
        )