from model_assisted_labeler.services.dataset_exporter import (
    CancellationCheck,
    DatasetExportRequest,
    DatasetExportResult,
    ProgressCallback,
)


class DatasetExportMixin:
    """Export the saved annotation pool as a YOLO dataset."""

    def export_dataset(
        self,
        request: DatasetExportRequest,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> DatasetExportResult:
        """Export the saved annotation pool as a YOLO dataset."""
        definition = self._require_definition()

        return self._dataset_exporter.export(
            definition=definition,
            request=request,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
        )
