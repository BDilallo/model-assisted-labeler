from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImageService:
    """
    Provides image discovery and image-dimension reading.

    This keeps filesystem and image-reading logic separate from the
    annotation session builder.
    """

    SUPPORTED_EXTENSIONS = {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }

    def discover_images(
        self,
        directory: Path,
        recursive: bool = False,
    ) -> list[Path]:
        """
        Find supported image files inside a directory.

        Args:
            directory:
                Directory to search.

            recursive:
                When True, search all subdirectories. When False,
                search only the supplied directory.

        Returns:
            Image paths sorted by relative path.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Image path is not a directory: {directory}"
            )

        if recursive:
            candidates = directory.rglob("*")
        else:
            candidates = directory.iterdir()

        image_paths = [
            path
            for path in candidates
            if (
                path.is_file()
                and path.suffix.casefold()
                in self.SUPPORTED_EXTENSIONS
            )
        ]

        image_paths.sort(
            key=lambda path: (
                path.relative_to(directory)
                .as_posix()
                .casefold()
            )
        )

        return image_paths

    def get_dimensions(
        self,
        image_path: Path,
    ) -> tuple[int, int]:
        """
        Read an image's width and height.

        Returns:
            A tuple containing:

                (width, height)
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Image path is not a file: {image_path}"
            )

        if (
            image_path.suffix.casefold()
            not in self.SUPPORTED_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}"
            )

        try:
            with Image.open(image_path) as image:
                width, height = image.size

        except UnidentifiedImageError as error:
            raise ValueError(
                f"File is not a readable image: {image_path}"
            ) from error

        except OSError as error:
            raise ValueError(
                f"Could not read image: {image_path}"
            ) from error

        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Image has invalid dimensions: {image_path}"
            )

        return width, height
