from typing import Tuple

from utils.data.manipulation.data_manipulator import DataManipulator


def determine_required_columns_multiple_data_manipulators(
    data_manipulators: Tuple[DataManipulator, ...]
) -> Tuple[str, ...]:
    # TODO(test): test it
    # TODO(medium): needs notion of active mode also in its usages and similar usages
    required_columns = set()
    added_columns_so_far = set()

    for data_manipulator in data_manipulators:
        added_columns_so_far.update(data_manipulator.added_columns())
        required_columns_not_added_yet = (
            set(data_manipulator.required_columns()) - added_columns_so_far
        )
        required_columns.update(required_columns_not_added_yet)

    required_columns = tuple(required_columns)

    return required_columns


def added_columns_multiple_data_manipulators(
    data_manipulators: Tuple[DataManipulator, ...]
) -> Tuple[str, ...]:
    added_columns = set()

    for data_manipulator in data_manipulators:
        added_columns.update(data_manipulator.added_columns())

    added_columns = tuple(added_columns)

    return added_columns
