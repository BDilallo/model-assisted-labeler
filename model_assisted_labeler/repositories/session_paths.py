from pathlib import Path

from model_assisted_labeler.models.session_definition import SessionDefinition

ANNOTATED_IMAGES_DIRECTORY = "Annotated Images"
ANNOTATIONS_DIRECTORY = "Annotations"
ANNOTATION_METADATA_DIRECTORY = "Annotation Metadata"


def annotations_directory(session_directory: Path) -> Path:
    return Path(session_directory) / ANNOTATIONS_DIRECTORY


def annotated_images_directory(session_directory: Path) -> Path:
    return Path(session_directory) / ANNOTATED_IMAGES_DIRECTORY


def annotation_metadata_directory(session_directory: Path) -> Path:
    return Path(session_directory) / ANNOTATION_METADATA_DIRECTORY


def validate_source_image_path(
    definition: SessionDefinition,
    image_path: Path,
) -> None:
    image_path = Path(image_path).resolve()
    source_directory = definition.image_directory.resolve()

    if image_path.parent != source_directory:
        raise ValueError(
            "Source images must be top-level files in the session "
            "image directory. Subdirectories are not used."
        )


def annotation_path_for(
    definition: SessionDefinition,
    image_path: Path,
) -> Path:
    validate_source_image_path(definition, image_path)
    return (
        annotations_directory(definition.session_directory)
        / Path(image_path).with_suffix(".txt").name
    )


def annotated_image_path_for(
    definition: SessionDefinition,
    image_path: Path,
) -> Path:
    validate_source_image_path(definition, image_path)
    return (
        annotated_images_directory(definition.session_directory)
        / Path(image_path).name
    )


def annotation_metadata_path_for(
    definition: SessionDefinition,
    image_path: Path,
) -> Path:
    validate_source_image_path(definition, image_path)
    return (
        annotation_metadata_directory(definition.session_directory)
        / f"{Path(image_path).stem}.json"
    )


def count_pooled_images(session_directory: Path) -> int:
    annotation_directory = annotations_directory(session_directory)
    image_directory = annotated_images_directory(session_directory)

    if not annotation_directory.is_dir() or not image_directory.is_dir():
        return 0

    copied_stems = {
        path.stem.casefold()
        for path in image_directory.iterdir()
        if path.is_file()
    }

    return sum(
        1
        for annotation_path in annotation_directory.glob("*.txt")
        if annotation_path.stem.casefold() in copied_stems
    )
