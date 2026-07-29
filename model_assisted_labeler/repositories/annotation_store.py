from pathlib import Path

from model_assisted_labeler.models.bounding_box import BoundingBox


class YoloAnnotationStore:
    """
    Loads and saves object-detection annotations using YOLO format.

    Each line in a YOLO label file uses:

        class_id center_x center_y width height

    Coordinate values are normalized between 0 and 1.
    """

    DECIMAL_PLACES = 6

    def load(
        self,
        label_path: Path,
        image_width: int,
        image_height: int,
    ) -> list[BoundingBox]:
        """
        Load bounding boxes from a YOLO label file.

        A missing label file is treated as an image with no annotations.
        """
        label_path = Path(label_path)

        self._validate_image_dimensions(
            image_width=image_width,
            image_height=image_height,
        )

        if not label_path.exists():
            return []

        if not label_path.is_file():
            raise ValueError(
                f"Label path is not a file: {label_path}"
            )

        annotations: list[BoundingBox] = []

        with label_path.open(
            mode="r",
            encoding="utf-8",
        ) as label_file:
            for line_number, line in enumerate(label_file, start=1):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                box = self._parse_line(
                    line=stripped_line,
                    line_number=line_number,
                    label_path=label_path,
                    image_width=image_width,
                    image_height=image_height,
                )

                annotations.append(box)

        return annotations

    def save(
        self,
        label_path: Path,
        annotations: list[BoundingBox],
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Save bounding boxes to a YOLO label file.

        Existing label contents are replaced. If the annotation list is
        empty, an empty label file is created.
        """
        label_path = Path(label_path)

        self._validate_image_dimensions(
            image_width=image_width,
            image_height=image_height,
        )

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_lines: list[str] = []

        for annotation_index, box in enumerate(annotations):
            if not isinstance(box, BoundingBox):
                raise TypeError(
                    "Annotations must contain only BoundingBox objects. "
                    f"Invalid item at index {annotation_index}."
                )

            box.normalize_coordinates()
            box.clamp(image_width, image_height)

            if not box.is_valid():
                raise ValueError(
                    f"Annotation at index {annotation_index} is invalid."
                )

            output_lines.append(
                self._format_box(
                    box=box,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        output_text = "\n".join(output_lines)

        if output_lines:
            output_text += "\n"

        temporary_path = label_path.with_suffix(
            label_path.suffix + ".tmp"
        )

        try:
            with temporary_path.open(
                mode="w",
                encoding="utf-8",
                newline="\n",
            ) as label_file:
                label_file.write(output_text)

            temporary_path.replace(label_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _parse_line(
        self,
        line: str,
        line_number: int,
        label_path: Path,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        """
        Convert one YOLO label line into a BoundingBox.
        """
        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid YOLO annotation in '{label_path}' "
                f"on line {line_number}: expected 5 values, "
                f"received {len(values)}."
            )

        try:
            class_id = int(values[0])
            center_x = float(values[1])
            center_y = float(values[2])
            width = float(values[3])
            height = float(values[4])

        except ValueError as error:
            raise ValueError(
                f"Invalid numeric value in '{label_path}' "
                f"on line {line_number}."
            ) from error

        self._validate_yolo_values(
            class_id=class_id,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            label_path=label_path,
            line_number=line_number,
        )

        box = BoundingBox.from_yolo(
            class_id=class_id,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            source="imported",
        )

        if not box.is_valid():
            raise ValueError(
                f"Annotation in '{label_path}' on line "
                f"{line_number} produced an invalid bounding box."
            )

        return box

    def _format_box(
        self,
        box: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> str:
        """
        Convert one BoundingBox into a YOLO label line.
        """
        (
            class_id,
            center_x,
            center_y,
            width,
            height,
        ) = box.to_yolo(
            image_width=image_width,
            image_height=image_height,
        )

        decimal_places = self.DECIMAL_PLACES

        return (
            f"{class_id} "
            f"{center_x:.{decimal_places}f} "
            f"{center_y:.{decimal_places}f} "
            f"{width:.{decimal_places}f} "
            f"{height:.{decimal_places}f}"
        )

    @staticmethod
    def _validate_image_dimensions(
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Ensure valid image dimensions were supplied.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Image dimensions must be greater than zero."
            )

    @staticmethod
    def _validate_yolo_values(
        class_id: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        label_path: Path,
        line_number: int,
    ) -> None:
        """
        Validate normalized values read from a YOLO label file.
        """
        if class_id < 0:
            raise ValueError(
                f"Negative class ID in '{label_path}' "
                f"on line {line_number}."
            )

        if not 0.0 <= center_x <= 1.0:
            raise ValueError(
                f"center_x is outside the normalized range in "
                f"'{label_path}' on line {line_number}."
            )

        if not 0.0 <= center_y <= 1.0:
            raise ValueError(
                f"center_y is outside the normalized range in "
                f"'{label_path}' on line {line_number}."
            )

        if not 0.0 < width <= 1.0:
            raise ValueError(
                f"Width is outside the normalized range in "
                f"'{label_path}' on line {line_number}."
            )

        if not 0.0 < height <= 1.0:
            raise ValueError(
                f"Height is outside the normalized range in "
                f"'{label_path}' on line {line_number}."
            )