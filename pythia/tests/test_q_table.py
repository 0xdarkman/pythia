import unittest

from pythia.reinforcement.q_function import InvalidAction
from pythia.reinforcement.q_table import QTable


class QTableTests(unittest.TestCase):
    def setUp(self):
        self.table = QTable([1, 2, 4])

    def test_creation(self):
        self.assertIsNotNone(self.table)

    def test_default_initial_value(self):
        self.assertAlmostEqual(0, self.table[[1, 2, 3], 1])

    def test_initial_value(self):
        t = QTable(range(0, 2), lambda: 1)
        self.assertAlmostEqual(1, t[[1, 0], 0])

    def test_update_value(self):
        self.table[[2, 3], 2] = 7
        self.table[[2, 3], 1] = 10
        self.assertAlmostEqual(10, self.table[[2, 3], 1])
        self.assertAlmostEqual(7, self.table[[2, 3], 2])
        self.table[[2, 3], 1] = 0
        self.assertAlmostEqual(0, self.table[[2, 3], 1])

    def test_scalar_state(self):
        self.table[3, 1] = 6
        self.assertAlmostEqual(6, self.table[3, 1])

    def test_try_set_action_out_of_action_space(self):
        with self.assertRaises(InvalidAction):
            unused = self.table[[2, 3], 0]

    def test_get_max_action_value_of_state(self):
        self.table[[3, 2, 1], 1] = -15.33
        self.table[[3, 2, 1], 2] = 298.521

        self.assertAlmostEqual(298.521, self.table.max_value_of([3, 2, 1]))

    def test_state_is_unaltered(self):
        s = [1, 2]
        self.table[s, 1] = 10.0
        self.assertSequenceEqual([1, 2], s)
        unused = self.table[s, 1]
        self.assertSequenceEqual([1, 2], s)
        self.assertSequenceEqual([1, 2], s)
        self.table.max_value_of(s)
        self.assertSequenceEqual([1, 2], s)


if __name__ == '__main__':
    unittest.main()
