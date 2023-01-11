from typing import Any

from misc.util import retain_some


def test_retain_some() -> None:

    def test_rs(
            input_arr: list[int],
            count: int,
            output_arr: list[int],
            delete_arr: set[int],
            **kwargs: Any) -> None:
        res, to_delete = retain_some(
            input_arr, count, key=lambda v: v, **kwargs)
        assert res == output_arr
        assert set(to_delete) == delete_arr

    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        4,
        [3, 3, 5, 6, 9],
        {0, 1, 2})
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        4,
        [3, 5, 6, 9],
        {0, 1, 2, 3},
        keep_last=False)
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        4,
        [6, 3, 2, 1, 0],
        {3, 5, 9},
        reverse=True)
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        4,
        [3, 2, 1, 0],
        {3, 5, 6, 9},
        reverse=True,
        keep_last=False)
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        5,
        [2, 3, 3, 5, 6, 9],
        {0, 1})
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        5,
        [3, 3, 5, 6, 9],
        {0, 1, 2},
        keep_last=False)
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        5,
        [6, 3, 3, 2, 1, 0],
        {5, 9},
        reverse=True)
    test_rs(
        [5, 3, 2, 1, 9, 3, 0, 6],
        5,
        [3, 3, 2, 1, 0],
        {5, 6, 9},
        reverse=True,
        keep_last=False)
    test_rs(
        [5, 3, 2, 1, 9, 3],
        5,
        [1, 2, 3, 3, 5, 9],
        set())
    test_rs(
        [5, 3, 2, 1, 9, 3],
        6,
        [1, 2, 3, 3, 5, 9],
        set(),
        keep_last=False)
    test_rs(
        [5, 3, 2],
        6,
        [2, 3, 5],
        set())
    test_rs(
        [5, 3, 2],
        6,
        [2, 3, 5],
        set(),
        keep_last=False)
