import io
import sys
import unittest
from contextlib import contextmanager

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


class ValueTableTests(unittest.TestCase):
    def setUp(self):
        self.table = ValueTable()

    def test_creation(self):
        self.assertIsNot(self.table, None)

    def test_retrieve_unvisited_state(self):
        t = ValueTable(lambda: 1)
        self.assertEqual(1, t.get_state([0, 1]))

    def test_update_state(self):
        self.table.update([1, 2], 10)
        self.assertEqual(10, self.table.get_state([1, 2]))

    def test_multiple_states(self):
        self.table.update([1, 2], 5)
        self.table.update([3, 6], 9)

        self.assertEqual(5, self.table.get_state([1, 2]))
        self.assertEqual(9, self.table.get_state([3, 6]))

    def test_update_existing_state(self):
        self.table.update([1.14, 2.159], 3)
        self.table.update([1.14, 2.159], 6)

        self.assertEqual(6, self.table.get_state([1.14, 2.159]))

    def test_index_operator(self):
        self.table[1, 0] = 20
        self.assertEqual(20, self.table[1, 0])

    def test_print_value_of_missing_state(self):
        with captured_output() as (out, err):
            self.table.print_states([[[1, 0]]])

        self.assertEqual("The state [1, 0] hasn't been set yet.", out.getvalue().strip())

    def test_print_values_of_one_missing_state(self):
        self.table.update([1, 0], 5)
        with captured_output() as (out, err):
            self.table.print_states([[[1, 0], [1, 3]]])

        self.assertEqual("The state [1, 3] hasn't been set yet.", out.getvalue().strip())

    def test_print_value_of_state(self):
        self.table.update([1, 0], 5)
        with captured_output() as (out, err):
            self.table.print_states([[[1, 0]]])

        self.assertEqual("{[1, 0], " + Formats.HIGHLIGHT + "  5.00" + Formats.END + "}", out.getvalue().strip())

    def test_print_values_of_multiple_state_sequences(self):
        self.table.update([1, 0], 5)
        self.table.update([2, 1], 10)
        self.table.update([1, 3], -1)
        with captured_output() as (out, err):
            self.table.print_states([[[1, 0], [1, 3]], [[2, 1]]])

        self.assertEqual(
            "{[1, 0], " + Formats.HIGHLIGHT + "  5.00" + Formats.END + "}, {[1, 3], " + Formats.REWARD + " -1.00" + Formats.END + "}\n{[2, 1], " + Formats.HIGHLIGHT + " 10.00" + Formats.END + "}",
            out.getvalue().strip())

    def test_print_values_float_formatting(self):
        self.table.update([1, 0], 12.313131313)
        with captured_output() as (out, err):
            self.table.print_states([[[1, 0]]])

        self.assertEqual("{[1, 0], " + Formats.HIGHLIGHT + " 12.31" + Formats.END + "}", out.getvalue().strip())

    def test_print_all(self):
        self.table[1.0, 2.0, 3] = 2
        self.table[1.0, 2.0, 2] = 1
        self.table[2.0, 2.0, 1] = 3
        self.table[2.0, 3.0, 3] = 5
        self.table[2.0, 3.0, -1] = 4
        with captured_output() as (out, err):
            self.table.print_all()

        self.assertEqual("[1.0 2.0 2] = 1\n"
                         "[1.0 2.0 3] = 2\n"
                         "[2.0 2.0 1] = 3\n"
                         "[2.0 3.0 -1] = 4\n"
                         "[2.0 3.0 3] = 5", out.getvalue().strip())

    def test_print_all_sorted(self):
        self.table[1.0, 2.0, 3] = 2
        self.table[1.0, 2.0, 2] = 1
        self.table[2.0, 2.0, 1] = 3
        self.table[2.0, 3.0, 4] = 5
        self.table[2.0, 3.0, 0] = 4
        with captured_output() as (out, err):
            self.table.print_all_sorted_by(2)

        self.assertEqual("[2.0 3.0 0] = 4\n"
                         "[2.0 2.0 1] = 3\n"
                         "[1.0 2.0 2] = 1\n"
                         "[1.0 2.0 3] = 2\n"
                         "[2.0 3.0 4] = 5", out.getvalue().strip())
