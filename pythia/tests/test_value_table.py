import io
import sys
from contextlib import contextmanager

import pytest

from pythia.reinforcement.value_table import ValueTable
from pythia.utils.ansi_formats import Formats


@contextmanager
def captured_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@pytest.fixture
def table():
    return ValueTable()


def test_initializer_value_at_unvisited_state():
    t = ValueTable(lambda: 1)
    assert 1 == t.get_state([0, 1])


def test_update_state(table):
    table.update([1, 2], 10)
    assert table.get_state([1, 2]) == 10


def test_multiple_states(table):
    table.update([1, 2], 5)
    table.update([3, 6], 9)

    assert table.get_state([1, 2]) == 5
    assert table.get_state([3, 6]) == 9


def test_update_existing_state(table):
    table.update([1.14, 2.159], 3)
    table.update([1.14, 2.159], 6)

    assert table.get_state([1.14, 2.159]) == 6


def test_index_operator(table):
    table[1, 0] = 20
    assert table[1, 0] == 20


def test_print_value_of_missing_state(table):
    with captured_output() as (out, err):
        table.print_states([[[1, 0]]])

    assert out.getvalue().strip() == "The state [1, 0] hasn't been set yet."


def test_print_values_of_one_missing_state(table):
    table.update([1, 0], 5)
    with captured_output() as (out, err):
        table.print_states([[[1, 0], [1, 3]]])

    assert out.getvalue().strip() == "The state [1, 3] hasn't been set yet."


def test_print_value_of_state(table):
    table.update([1, 0], 5)
    with captured_output() as (out, err):
        table.print_states([[[1, 0]]])

    assert out.getvalue().strip() == "{[1, 0], " + Formats.HIGHLIGHT + "  5.00" + Formats.END + "}"


def test_print_values_of_multiple_state_sequences(table):
    table.update([1, 0], 5)
    table.update([2, 1], 10)
    table.update([1, 3], -1)
    with captured_output() as (out, err):
        table.print_states([[[1, 0], [1, 3]], [[2, 1]]])

    assert out.getvalue().strip() == \
           "{[1, 0], " + Formats.HIGHLIGHT + "  5.00" + Formats.END + "}, {[1, 3], " + Formats.REWARD + " -1.00" + Formats.END + "}\n{[2, 1], " + Formats.HIGHLIGHT + " 10.00" + Formats.END + "}"


def test_print_values_float_formatting(table):
    table.update([1, 0], 12.313131313)
    with captured_output() as (out, err):
        table.print_states([[[1, 0]]])

    assert out.getvalue().strip() == "{[1, 0], " + Formats.HIGHLIGHT + " 12.31" + Formats.END + "}"


def test_print_all(table):
    table[1.0, 2.0, 3] = 2
    table[1.0, 2.0, 2] = 1
    table[2.0, 2.0, 1] = 3
    table[2.0, 3.0, 3] = 5
    table[2.0, 3.0, -1] = 4
    with captured_output() as (out, err):
        table.print_all()

    assert out.getvalue().strip() == \
           "[1.0 2.0 2] = 1\n" \
           "[1.0 2.0 3] = 2\n" \
           "[2.0 2.0 1] = 3\n" \
           "[2.0 3.0 -1] = 4\n" \
           "[2.0 3.0 3] = 5"


def test_print_all_sorted(table):
    table[1.0, 2.0, 3] = 2
    table[1.0, 2.0, 2] = 1
    table[2.0, 2.0, 1] = 3
    table[2.0, 3.0, 4] = 5
    table[2.0, 3.0, 0] = 4
    with captured_output() as (out, err):
        table.print_all_sorted_by(2)

    assert out.getvalue().strip() == \
           "[2.0 3.0 0] = 4\n" \
           "[2.0 2.0 1] = 3\n" \
           "[1.0 2.0 2] = 1\n" \
           "[1.0 2.0 3] = 2\n" \
           "[2.0 3.0 4] = 5"
