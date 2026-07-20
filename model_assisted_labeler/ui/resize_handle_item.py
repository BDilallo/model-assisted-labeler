from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
)

from model_assisted_labeler.geometry.bounding_box_geometry import (
    ResizeHandle,
)

if TYPE_CHECKING:
    from model_assisted_labeler.ui.bounding_box_item import BoundingBoxItem


class ResizeHandleItem(QGraphicsEllipseItem):
    """
    Displays one visible corner control for a bounding box.

    The item ignores view scaling so its on-screen diameter remains the
    same while the image is zoomed in, zoomed out, or fitted to the view.
    """

    DIAMETER = 15.0
    BORDER_WIDTH = 4.0

    def __init__(
        self,
        handle: ResizeHandle,
        color: QColor,
        parent: QGraphicsItem,
    ) -> None:
        super().__init__(parent)

        self._handle = handle
        self._normal_fill = QColor(255, 255, 255)
        self._hover_fill = QColor(255, 225, 80)

        radius = self.DIAMETER / 2.0

        self.setRect(
            QRectF(
                -radius,
                -radius,
                self.DIAMETER,
                self.DIAMETER,
            )
        )

        handle_pen = QPen(color)
        handle_pen.setWidthF(self.BORDER_WIDTH)
        handle_pen.setCosmetic(True)

        self.setPen(handle_pen)
        self.setBrush(self._normal_fill)

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )

        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setZValue(100.0)
        self.setVisible(False)
        self.setToolTip("Drag this control to resize the box")

        if handle in {
            ResizeHandle.TOP_LEFT,
            ResizeHandle.BOTTOM_RIGHT,
        }:
            cursor = Qt.CursorShape.SizeFDiagCursor
        else:
            cursor = Qt.CursorShape.SizeBDiagCursor

        self.setCursor(QCursor(cursor))

    @property
    def handle(self) -> ResizeHandle:
        """Return the corner represented by this control."""
        return self._handle

    def set_color(
        self,
        color: QColor,
    ) -> None:
        """Update the outline color used by the resize control."""
        handle_pen = QPen(color)
        handle_pen.setWidthF(self.BORDER_WIDTH)
        handle_pen.setCosmetic(True)
        self.setPen(handle_pen)
        self.update()

    def hoverEnterEvent(
        self,
        event: QGraphicsSceneHoverEvent,
    ) -> None:
        """Highlight the control under the pointer."""
        self.setBrush(self._hover_fill)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(
        self,
        event: QGraphicsSceneHoverEvent,
    ) -> None:
        """Restore the control's normal appearance."""
        self.setBrush(self._normal_fill)
        super().hoverLeaveEvent(event)

    def mousePressEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """Begin resizing the parent bounding box."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        parent_box = self._parent_box()
        parent_box.begin_resize(self._handle)

        event.accept()

    def mouseMoveEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """Continue resizing toward the current scene position."""
        parent_box = self._parent_box()
        parent_box.continue_resize(event.scenePos())

        event.accept()

    def mouseReleaseEvent(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> None:
        """Finish the active resize operation."""
        parent_box = self._parent_box()
        parent_box.finish_resize()

        event.accept()

    def _parent_box(self) -> BoundingBoxItem:
        """Return the BoundingBoxItem that owns this handle."""
        parent_item = self.parentItem()

        if parent_item is None:
            raise RuntimeError(
                "ResizeHandleItem must have a BoundingBoxItem parent."
            )

        return cast("BoundingBoxItem", parent_item)
