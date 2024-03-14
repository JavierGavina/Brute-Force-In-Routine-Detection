import unittest
import sys

sys.path.append('..')

import datetime
from src.structures import Subsequence, Sequence, Cluster, Routines
import numpy as np

date = datetime.datetime.now()

subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.datetime.now(), 0)
subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.datetime.now(), 1)
subsequence3 = Subsequence(np.array([9, 10, 11, 12]), datetime.datetime.now(), 2)
subsequence4 = Subsequence(np.array([13, 14, 15, 16]), datetime.datetime.now(), 3)


class TestSubsequence(unittest.TestCase):
    def test_init(self):
        instance = np.array([1, 2, 3, 4])
        starting_point = 0
        self.assertEqual(np.array_equal(subsequence1.get_instance(), instance), True)
        self.assertEqual(subsequence1.get_date(), date)
        self.assertEqual(subsequence1.get_starting_point(), starting_point)

    def test_Distance(self):
        instance = np.array([1, 2, 3, 4])
        starting_point = 0
        subsequence = Subsequence(instance, date, starting_point)
        self.assertEqual(subsequence.Distance(subsequence), 0)

    def test_Magnitude(self):
        self.assertEqual(subsequence1.Magnitude(), 4)
        self.assertEqual(subsequence2.Magnitude(), 8)
        self.assertEqual(subsequence3.Magnitude(), 12)
        self.assertEqual(subsequence4.Magnitude(), 16)

    def test__len__(self):
        self.assertEqual(len(subsequence1), 4)
    def test__getitem__(self):
        self.assertEqual(subsequence1[0], 1)
        self.assertEqual(subsequence1[1], 2)
        self.assertEqual(subsequence1[2], 3)
        self.assertEqual(subsequence1[3], 4)

    def test__eq__(self):

        # CASE 1: True
        other = Subsequence(np.array([1, 2, 3, 4]), date, 0)
        self.assertEqual(subsequence1 == other, True)

        # CASE 2: False
        other = Subsequence(np.array([1, 2, 3, 4]), date, 1)
        self.assertEqual(subsequence1 == other, False)

        # CASE 3: False
        other = Subsequence(np.array([1, 2, 5, 4]), date, 0)
        self.assertEqual(subsequence1 == other, False)

        # CASE 4: False
        other = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2024, 1, 1), 0)
        self.assertEqual(subsequence1 == other, False)


class TestSequence(unittest.TestCase):
    def test_init(self):
        sequence = Sequence()
        self.assertEqual(len(sequence), 0)

    def test_add_sequence(self):
        sequence = Sequence(subsequence=subsequence1)
        sequence.add_sequence(subsequence2)
        self.assertEqual(len(sequence), 2)

    def test_get_by_starting_point(self):
        sequence = Sequence(subsequence1)

        sequence.add_sequence(subsequence2)
        sequence.add_sequence(subsequence3)
        sequence.add_sequence(subsequence4)

        # check if returns the correct subsequence
        self.assertEqual(np.array_equal(sequence.get_by_starting_point(2).get_instance(),
                                        np.array([9, 10, 11, 12])),
                         True, "Should return [9, 10, 11, 12]")

        # check if returns none
        self.assertEqual(sequence.get_by_starting_point(5), None, "Should return None")

    def test__getitem__(self):
        sequence = Sequence(subsequence1)
        self.assertEqual(np.array_equal(sequence[0].get_instance(), np.array([1, 2, 3, 4])), True)

    def test__iter__(self):
        sequence = Sequence(subsequence1)
        for subsequence in sequence:
            self.assertEqual(np.array_equal(subsequence.get_instance(), np.array([1, 2, 3, 4])), True)


if __name__ == '__main__':
    unittest.main()
