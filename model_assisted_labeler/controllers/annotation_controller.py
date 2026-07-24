from model_assisted_labeler.controllers._base import (
    _AnnotationControllerBase,
)
from model_assisted_labeler.controllers._classes import (
    ClassManagementMixin,
)
from model_assisted_labeler.controllers._editing import (
    AnnotationEditingMixin,
)
from model_assisted_labeler.controllers._export import (
    DatasetExportMixin,
)
from model_assisted_labeler.controllers._lifecycle import (
    SessionLifecycleMixin,
)
from model_assisted_labeler.controllers._model_management import (
    ModelManagementMixin,
)
from model_assisted_labeler.controllers._prediction import (
    PredictionMixin,
)

__all__ = ["AnnotationController"]


class AnnotationController(
    ModelManagementMixin,
    SessionLifecycleMixin,
    PredictionMixin,
    AnnotationEditingMixin,
    ClassManagementMixin,
    DatasetExportMixin,
    _AnnotationControllerBase,
):
    """Coordinate session state, persistence, and model prediction."""
