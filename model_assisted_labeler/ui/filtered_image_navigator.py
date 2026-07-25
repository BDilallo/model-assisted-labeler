from model_assisted_labeler.controllers.application_controller import (
    AnnotationController,
)
from model_assisted_labeler.ui.image_filter_bar import ImageFilterBar


class FilteredImageNavigator:
    """
    Track which session images match the active filter-bar controls.

    Drives Back/Next navigation, filtered position text, and the
    nearby-annotation prefetch window without duplicating the
    filter-matching logic already owned by the controller.
    """

    def __init__(
        self,
        controller: AnnotationController,
        filter_bar: ImageFilterBar,
    ) -> None:
        self._controller = controller
        self._filter_bar = filter_bar
        self._matching_indexes: list[int] = []

    @property
    def matching_indexes(self) -> list[int]:
        return self._matching_indexes

    def contains(self, index: int | None) -> bool:
        return index in self._matching_indexes

    def rebuild(self) -> None:
        """Recompute the full set of images matching the active filter."""
        session = self._controller.session

        if session is None:
            self._matching_indexes = []
            self._filter_bar.set_result_count(0, 0)
            return

        self._matching_indexes = self._controller.image_indexes_matching(
            filter_key=self._filter_bar.filter_key,
            confidence_threshold=self._filter_bar.confidence_threshold,
            class_id=self._filter_bar.class_id,
        )
        self._filter_bar.set_result_count(
            len(self._matching_indexes),
            session.image_count,
        )

    def refresh_current_membership(self) -> None:
        """Incrementally update membership for only the current image."""
        session = self._controller.session

        if session is None or not session.images:
            self._matching_indexes = []
            self._filter_bar.set_result_count(0, 0)
            return

        current_index = session.current_index
        matches = self._controller.image_index_matches_filter(
            image_index=current_index,
            filter_key=self._filter_bar.filter_key,
            confidence_threshold=self._filter_bar.confidence_threshold,
            class_id=self._filter_bar.class_id,
        )
        currently_in_filter = current_index in self._matching_indexes

        if matches and not currently_in_filter:
            self._matching_indexes.append(current_index)
            self._matching_indexes.sort()
        elif not matches and currently_in_filter:
            self._matching_indexes.remove(current_index)

        self._filter_bar.set_result_count(
            len(self._matching_indexes),
            session.image_count,
        )

    def previous_matching_index(self) -> int | None:
        session = self._controller.session

        if session is None:
            return None

        previous_indexes = [
            index
            for index in self._matching_indexes
            if index < session.current_index
        ]

        if not previous_indexes:
            return None

        return previous_indexes[-1]

    def next_matching_index(self) -> int | None:
        session = self._controller.session

        if session is None:
            return None

        return next(
            (
                index
                for index in self._matching_indexes
                if index > session.current_index
            ),
            None,
        )

    def current_position(self) -> int | None:
        session = self._controller.session

        if session is None:
            return None

        try:
            return (
                self._matching_indexes.index(session.current_index) + 1
            )
        except ValueError:
            return None

    def prefetch_window(self, radius: int) -> list[int]:
        """Return the indexes to prefetch around the current image."""
        session = self._controller.session

        if session is None or not session.images:
            return []

        current_index = session.current_index

        if current_index in self._matching_indexes:
            filtered_position = self._matching_indexes.index(
                current_index
            )
            start_position = max(0, filtered_position - radius)
            end_position = min(
                len(self._matching_indexes),
                filtered_position + radius + 1,
            )
            return self._matching_indexes[start_position:end_position]

        return [current_index]
