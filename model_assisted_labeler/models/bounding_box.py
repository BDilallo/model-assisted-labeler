from dataclasses import dataclass


@dataclass
class BoundingBox:
    """
    Represents one rectangular object annotation.

    Coordinates are stored in image pixels using the xyxy format:

        x1, y1 = top-left corner
        x2, y2 = bottom-right corner
    """

    class_id: int

    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float | None = None
    source: str = "manual"

    @property
    def width(self) -> float:
        """Return the width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return the height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Return the area of the bounding box in square pixels."""
        return self.width * self.height

    def is_valid(self) -> bool:
        """
        Return True when the bounding box has a valid class ID,
        width, and height.
        """
        return (
            self.class_id >= 0
            and self.width > 0
            and self.height > 0
        )

    def normalize_coordinates(self) -> None:
        """
        Ensure x1/y1 represent the top-left corner and
        x2/y2 represent the bottom-right corner.

        This is useful when a user draws a box in reverse,
        such as dragging from bottom-right to top-left.
        """
        left = min(self.x1, self.x2)
        right = max(self.x1, self.x2)
        top = min(self.y1, self.y2)
        bottom = max(self.y1, self.y2)

        self.x1 = left
        self.x2 = right
        self.y1 = top
        self.y2 = bottom

    def clamp(
        self,
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Keep the bounding box inside the image boundaries.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be greater than zero.")

        self.normalize_coordinates()

        self.x1 = max(0.0, min(self.x1, float(image_width)))
        self.y1 = max(0.0, min(self.y1, float(image_height)))
        self.x2 = max(0.0, min(self.x2, float(image_width)))
        self.y2 = max(0.0, min(self.y2, float(image_height)))

    def to_yolo(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[int, float, float, float, float]:
        """
        Convert the pixel-based box into normalized YOLO format.

        Returns:
            (
                class_id,
                normalized_center_x,
                normalized_center_y,
                normalized_width,
                normalized_height,
            )
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be greater than zero.")

        if not self.is_valid():
            raise ValueError("Cannot convert an invalid bounding box.")

        center_x = self.x1 + (self.width / 2)
        center_y = self.y1 + (self.height / 2)

        normalized_center_x = center_x / image_width
        normalized_center_y = center_y / image_height
        normalized_width = self.width / image_width
        normalized_height = self.height / image_height

        return (
            self.class_id,
            normalized_center_x,
            normalized_center_y,
            normalized_width,
            normalized_height,
        )

    @classmethod
    def from_yolo(
        cls,
        class_id: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
        source: str = "imported",
    ) -> "BoundingBox":
        """
        Create a pixel-based BoundingBox from normalized YOLO values.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be greater than zero.")

        pixel_center_x = center_x * image_width
        pixel_center_y = center_y * image_height
        pixel_width = width * image_width
        pixel_height = height * image_height

        x1 = pixel_center_x - (pixel_width / 2)
        y1 = pixel_center_y - (pixel_height / 2)
        x2 = pixel_center_x + (pixel_width / 2)
        y2 = pixel_center_y + (pixel_height / 2)

        box = cls(
            class_id=class_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=None,
            source=source,
        )

        box.clamp(image_width, image_height)

        return box