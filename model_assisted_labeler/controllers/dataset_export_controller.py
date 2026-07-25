from model_assisted_labeler.controllers.session_context import SessionContext
from model_assisted_labeler.models.image_record import ImageRecord
from model_assisted_labeler.services.dataset_export_service import (
    CancellationCallback,
    DatasetExportResult,
    DatasetExportService,
    DatasetExportSettings,
    ProgressCallback,
)


class DatasetExportController:
    """Export the session's saved annotations as a ready-to-train dataset."""

    def __init__(
        self,
        context: SessionContext,
        export_service: DatasetExportService,
    ) -> None:
        self._context = context
        self._export_service = export_service

    @property
    def pooled_image_count(self) -> int:
        session = self._context.session

        if session is None:
            return 0

        return len(self._pooled_images(session.images))

    def compute_export_split(
        self,
        settings: DatasetExportSettings,
    ) -> tuple[int, int]:
        return self._export_service.compute_split_counts(
            self.pooled_image_count,
            settings,
        )

    def export_dataset(
        self,
        settings: DatasetExportSettings,
        progress_callback: ProgressCallback | None = None,
        cancellation_requested: CancellationCallback | None = None,
    ) -> DatasetExportResult:
        session = self._context.require_session()
        definition = self._context.require_definition()

        return self._export_service.export(
            definition=definition,
            pooled_images=self._pooled_images(session.images),
            classes=session.classes,
            settings=settings,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
        )

    @staticmethod
    def _pooled_images(
        images: list[ImageRecord],
    ) -> list[ImageRecord]:
        return [
            image_record
            for image_record in images
            if image_record.in_annotation_pool
        ]
