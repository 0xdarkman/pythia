import pytest

from pythia.core.environment.stock_markov_process import StockMarkovProcess
from pythia.core.environment.stocks import StockData
from .common_fixtures import model_path


@pytest.fixture
def process():
    return StockMarkovProcess(StockData(model_path()))


def test_first_step_states(process):
    states = process.get_states_of_step(0)
    assert states == [[3, 0, 0, 0]]


def test_second_step_states(process):
    states = process.get_states_of_step(1)
    assert states == [[5, 0, 1, 0], [5, 3, 1, 0]]


def test_third_step_states(process):
    states = process.get_states_of_step(2)
    assert states == [[4, 0, 2, 0], [4, 0, 2, 2], [4, 3, 2, 0], [4, 5, 2, 0]]


def test_fourth_step_states(process):
    states = process.get_states_of_step(3)

    assert states == [[1, 0, 3, 0], [1, 0, 3, 2], [1, 4, 3, 2], [1, 0, 3, 1], [1, 0, 3, -1], [1, 3, 3, 0], [1, 5, 3, 0],
                      [1, 4, 3, 0]]
