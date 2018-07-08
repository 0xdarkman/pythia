import pandas as pd
import pytest

from pythia.core.streams.data_frame_stream import DataFrameStream, Exchange


def make_stream_with_one_item():
    return DataFrameStream(pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [1], "volume": [10]}), name="SYMA")


def test_next_gives_next_row_in_frame():
    s = DataFrameStream(pd.DataFrame({"open": [1, 2], "high": [2, 4], "low": [1, 1], "close": [1, 3], "volume": [10, 10]}),
                        name="SYMA")
    assert next(s) == {"CASH_SYMA": Exchange(1, 2, 1, 1, 10), "SYMA_CASH": Exchange(1, 1 / 2, 1, 1, 10)}
    assert next(s) == {"CASH_SYMA": Exchange(2, 4, 1, 3, 10), "SYMA_CASH": Exchange(1 / 2, 1 / 4, 1, 1 / 3, 10)}


def test_raises_stop_iteration_when_iterating_passed_data():
    with pytest.raises(StopIteration):
        s = make_stream_with_one_item()
        next(s)
        next(s)


def test_return_default_value_when_iterating_passed_data_and_default_has_been_specified():
    s = make_stream_with_one_item()
    next(s)
    assert next(s, None) is None


def test_reset_series():
    s = make_stream_with_one_item()
    next(s)
    s.reset()
    assert next(s) == {"CASH_SYMA": Exchange(1, 2, 1, 1, 10), "SYMA_CASH": Exchange(1, 1 / 2, 1, 1, 10)}
