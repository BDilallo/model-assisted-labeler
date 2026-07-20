from pathlib import Path
from typing import Protocol


from model_assisted_labeler.models.bounding_box import BoundingBox


class DetectionModelRunner(Protocol):
    """
    Defines the behavior required from an object-detection runner.

    Any model runner used by the application must be able to load a
    model and convert its predictions into BoundingBox objects.
    """

    @property
    def is_loaded(self) -> bool:
        """Return True when a detection model is loaded."""
        ...

    @property
    def model_path(self) -> Path | None:
        """Return the path of the currently loaded model."""
        ...

    @property
    def class_names(self) -> dict[int, str]:
        """Return the model's class ID-to-name mapping."""
        ...

    def load_model(
        self,
        model_path: Path,
    ) -> None:
        """Load a detection model from a local file."""
        ...

    def unload_model(self) -> None:
        """Unload the current model."""
        ...

    def predict(
        self,
        image_path: Path,
    ) -> list[BoundingBox]:
        """Run inference and return detected bounding boxes."""
        ...


class UltralyticsDetectionRunner:
    """
    Runs object detection using an Ultralytics YOLO model.

    Ultralytics-specific results are converted into the application's
    BoundingBox representation before they leave this class.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        """
        Initialize the runner without immediately loading a model.

        Args:
            confidence_threshold:
                Predictions below this confidence are excluded.

            device:
                Device passed to Ultralytics for inference. Examples
                include "cpu", 0, or None for automatic selection.
        """
        self._model: object | None = None
        self._model_path: Path | None = None
        self._confidence_threshold = 0.25
        self._device = device

        self.set_confidence_threshold(confidence_threshold)

    @property
    def is_loaded(self) -> bool:
        """Return True when a model is currently loaded."""
        return self._model is not None

    @property
    def model_path(self) -> Path | None:
        """Return the path of the currently loaded model."""
        return self._model_path

    @property
    def confidence_threshold(self) -> float:
        """Return the current prediction confidence threshold."""
        return self._confidence_threshold

    @property
    def class_names(self) -> dict[int, str]:
        """
        Return the loaded model's class ID-to-name mapping.

        Returns an empty dictionary when no model is loaded.
        """
        if self._model is None:
            return {}

        model_names = self._model.names

        if isinstance(model_names, dict):
            return {
                int(class_id): str(class_name)
                for class_id, class_name in model_names.items()
            }

        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(model_names)
        }

    def set_confidence_threshold(
        self,
        confidence_threshold: float,
    ) -> None:
        """
        Set the minimum confidence required for predictions.

        The value must be between zero and one.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0 and 1."
            )

        self._confidence_threshold = float(
            confidence_threshold
        )

    def load_model(
        self,
        model_path: Path,
    ) -> None:
        """
        Load an Ultralytics object-detection model.

        Loading another model replaces the previously loaded model.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file does not exist: {model_path}"
            )

        if not model_path.is_file():
            raise ValueError(
                f"Model path is not a file: {model_path}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required to load a detection model. "
                "Install it in the active virtual environment first."
            ) from error

        try:
            loaded_model = YOLO(
                str(model_path),
                task="detect",
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not load detection model: {model_path}"
            ) from error

        self._model = loaded_model
        self._model_path = model_path

    def unload_model(self) -> None:
        """Unload the currently active model."""
        self._model = None
        self._model_path = None

    def predict(
        self,
        image_path: Path,
    ) -> list[BoundingBox]:
        """
        Run object detection on one image.

        Returns:
            A list of BoundingBox objects using original-image pixel
            coordinates.
        """
        if self._model is None:
            raise RuntimeError(
                "A detection model must be loaded before prediction."
            )

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Image path is not a file: {image_path}"
            )

        try:
            results = self._model.predict(
                source=str(image_path),
                conf=self._confidence_threshold,
                device=self._device,
                verbose=False,
            )

        except Exception as error:
            raise RuntimeError(
                f"Model prediction failed for: {image_path}"
            ) from error

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            raise RuntimeError(
                "The loaded model did not return standard "
                "object-detection bounding boxes."
            )

        image_height, image_width = result.orig_shape

        coordinate_rows = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidence_scores = result.boxes.conf.cpu().tolist()

        if not (
            len(coordinate_rows)
            == len(class_ids)
            == len(confidence_scores)
        ):
            raise RuntimeError(
                "Model result fields contain mismatched lengths."
            )

        predictions: list[BoundingBox] = []

        for coordinates, class_id, confidence in zip(
            coordinate_rows,
            class_ids,
            confidence_scores,
        ):
            if len(coordinates) != 4:
                raise RuntimeError(
                    "Model returned an invalid bounding-box shape."
                )

            x1, y1, x2, y2 = coordinates

            box = BoundingBox(
                class_id=int(class_id),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(confidence),
                source="model",
            )

            box.normalize_coordinates()
            box.clamp(
                image_width=int(image_width),
                image_height=int(image_height),
            )

            if box.is_valid():
                predictions.append(box)

        return predictions