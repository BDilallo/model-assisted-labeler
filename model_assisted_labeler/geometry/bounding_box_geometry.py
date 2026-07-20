from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import isclose, isfinite


class ResizeHandle(Enum):
    """Identifies the corner used to resize a rectangle."""

    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


@dataclass(frozen=True)
class BoundingBoxGeometry:
    """
    Stores rectangular geometry without depending on a GUI framework.

    The values use image coordinates, where left/top represent the
    upper-left corner and right/bottom represent the lower-right corner.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (
            self.left,
            self.top,
            self.right,
            self.bottom,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError(
                "Bounding-box geometry must contain finite values."
            )

        if self.right < self.left:
            raise ValueError(
                "Geometry right edge cannot be left of its left edge."
            )

        if self.bottom < self.top:
            raise ValueError(
                "Geometry bottom edge cannot be above its top edge."
            )

    @classmethod
    def from_position_and_size(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> BoundingBoxGeometry:
        """Create geometry from an upper-left position and size."""
        if width < 0.0 or height < 0.0:
            raise ValueError(
                "Geometry width and height cannot be negative."
            )

        return cls(
            left=float(x),
            top=float(y),
            right=float(x + width),
            bottom=float(y + height),
        )

    @property
    def width(self) -> float:
        """Return the rectangle width."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Return the rectangle height."""
        return self.bottom - self.top

    def is_valid(
        self,
        minimum_size: float = 0.0,
    ) -> bool:
        """Return True when both dimensions exceed the minimum size."""
        if minimum_size < 0.0:
            raise ValueError("Minimum size cannot be negative.")

        return (
            self.width >= minimum_size
            and self.height >= minimum_size
            and self.width > 0.0
            and self.height > 0.0
        )

    def clamp_to_bounds(
        self,
        bounds: BoundingBoxGeometry,
    ) -> BoundingBoxGeometry:
        """Clip every edge so the rectangle remains inside bounds."""
        return BoundingBoxGeometry(
            left=min(max(self.left, bounds.left), bounds.right),
            top=min(max(self.top, bounds.top), bounds.bottom),
            right=min(max(self.right, bounds.left), bounds.right),
            bottom=min(max(self.bottom, bounds.top), bounds.bottom),
        )

    def move_to_clamped(
        self,
        x: float,
        y: float,
        bounds: BoundingBoxGeometry,
    ) -> BoundingBoxGeometry:
        """
        Move the rectangle while keeping its complete area inside bounds.
        """
        maximum_x = bounds.right - self.width
        maximum_y = bounds.bottom - self.height

        clamped_x = min(
            max(float(x), bounds.left),
            maximum_x,
        )

        clamped_y = min(
            max(float(y), bounds.top),
            maximum_y,
        )

        return BoundingBoxGeometry.from_position_and_size(
            x=clamped_x,
            y=clamped_y,
            width=self.width,
            height=self.height,
        )

    def resize_from_handle(
        self,
        handle: ResizeHandle,
        pointer_x: float,
        pointer_y: float,
        bounds: BoundingBoxGeometry,
        minimum_size: float,
    ) -> BoundingBoxGeometry:
        """
        Resize one corner toward a pointer position.

        The opposite corner remains fixed. The result stays inside the
        supplied bounds and cannot become smaller than minimum_size.
        """
        if minimum_size <= 0.0:
            raise ValueError(
                "Minimum resize size must be greater than zero."
            )

        left = self.left
        top = self.top
        right = self.right
        bottom = self.bottom

        pointer_x = min(
            max(float(pointer_x), bounds.left),
            bounds.right,
        )

        pointer_y = min(
            max(float(pointer_y), bounds.top),
            bounds.bottom,
        )

        if handle == ResizeHandle.TOP_LEFT:
            left = min(pointer_x, right - minimum_size)
            top = min(pointer_y, bottom - minimum_size)

        elif handle == ResizeHandle.TOP_RIGHT:
            right = max(pointer_x, left + minimum_size)
            top = min(pointer_y, bottom - minimum_size)

        elif handle == ResizeHandle.BOTTOM_LEFT:
            left = min(pointer_x, right - minimum_size)
            bottom = max(pointer_y, top + minimum_size)

        elif handle == ResizeHandle.BOTTOM_RIGHT:
            right = max(pointer_x, left + minimum_size)
            bottom = max(pointer_y, top + minimum_size)

        else:
            raise ValueError(f"Unsupported resize handle: {handle}")

        return BoundingBoxGeometry(
            left=max(left, bounds.left),
            top=max(top, bounds.top),
            right=min(right, bounds.right),
            bottom=min(bottom, bounds.bottom),
        )

    def is_close_to(
        self,
        other: BoundingBoxGeometry,
        tolerance: float = 1e-6,
    ) -> bool:
        """Return True when two rectangles are effectively identical."""
        if tolerance < 0.0:
            raise ValueError("Tolerance cannot be negative.")

        return all(
            isclose(first, second, abs_tol=tolerance)
            for first, second in zip(
                (
                    self.left,
                    self.top,
                    self.right,
                    self.bottom,
                ),
                (
                    other.left,
                    other.top,
                    other.right,
                    other.bottom,
                ),
            )
        )
