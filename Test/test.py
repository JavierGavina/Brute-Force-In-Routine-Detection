import unittest
import sys

sys.path.append('..')

import datetime
from src.structures import Subsequence, Sequence, Cluster, Routines
import numpy as np


class TestSubsequence(unittest.TestCase):
    def setUp(self):
        self.date = datetime.date(2021, 1, 1)
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.subsequence3 = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)
        self.subsequence4 = Subsequence(np.array([13, 14, 15, 16]), datetime.date(2021, 1, 4), 12)


    def test_init(self):
        instance = np.array([1, 2, 3, 4])
        starting_point = 0
        self.assertEqual(np.array_equal(self.subsequence1.get_instance(), instance), True)
        self.assertEqual(self.subsequence1.get_date(), self.date)
        self.assertEqual(self.subsequence1.get_starting_point(), starting_point)

    def test_Distance(self):
        instance = np.array([1, 2, 3, 4])
        starting_point = 0
        subsequence = Subsequence(instance, self.date, starting_point)
        self.assertEqual(subsequence.Distance(subsequence), 0)

    def test_Magnitude(self):
        self.assertEqual(self.subsequence1.Magnitude(), 4)
        self.assertEqual(self.subsequence2.Magnitude(), 8)
        self.assertEqual(self.subsequence3.Magnitude(), 12)
        self.assertEqual(self.subsequence4.Magnitude(), 16)

    def test__len__(self):
        self.assertEqual(len(self.subsequence1), 4)

    def test__getitem__(self):
        self.assertEqual(self.subsequence1[0], 1)
        self.assertEqual(self.subsequence1[1], 2)
        self.assertEqual(self.subsequence1[2], 3)
        self.assertEqual(self.subsequence1[3], 4)

    def test__eq__(self):
        # CASE 1: True
        other = Subsequence(np.array([1, 2, 3, 4]), self.date, 0)
        self.assertEqual(self.subsequence1 == other, True)

        # CASE 2: False
        other = Subsequence(np.array([1, 2, 3, 4]), self.date, 1)
        self.assertEqual(self.subsequence1 == other, False)

        # CASE 3: False
        other = Subsequence(np.array([1, 2, 5, 4]), self.date, 0)
        self.assertEqual(self.subsequence1 == other, False)

        # CASE 4: False
        other = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2024, 1, 1), 0)
        self.assertEqual(self.subsequence1 == other, False)


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

    def test_add_sequence(self):
        self.sequence.add_sequence(self.subsequence1)
        self.assertEqual(len(self.sequence), 1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(len(self.sequence), 2)

    def test_get_by_starting_point(self):
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_by_starting_point(0), self.subsequence1)
        self.assertEqual(self.sequence.get_by_starting_point(4), self.subsequence2)

    def test_get_starting_points(self):
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_starting_points(), [0, 4])

    def test_get_dates(self):
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_dates(), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)])

    def test_get_subsequences(self):
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(np.array_equal(np.array(self.sequence.get_subsequences()), np.array([np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])])), True, msg="Expected: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])], Got: " + str(self.sequence.get_subsequences()))


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.cluster = Cluster(centroid=np.array([10, 10, 10, 10]), instances=self.sequence)

    def test_add_instance(self):
        new_subsequence = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)
        self.cluster.add_instance(new_subsequence)
        self.assertEqual(len(self.cluster.get_sequences()), 3)

    def test_get_sequences(self):
        self.assertEqual(self.cluster.get_sequences(), self.sequence)

    def test_update_centroid(self):
        self.cluster.update_centroid()
        self.assertTrue(np.array_equal(self.cluster.centroid, np.array([3.0, 4.0, 5.0, 6.0])))

    def test_get_starting_points(self):
        self.assertEqual(self.cluster.get_starting_points(), [0, 4])

    def test_get_dates(self):
        self.assertEqual(self.cluster.get_dates(), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)])

    def test_centroid_getter(self):
        self.assertTrue(np.array_equal(self.cluster.centroid, np.array([10, 10, 10, 10])))

    def test_centroid_setter(self):
        new_centroid = np.array([1, 2, 3, 4])
        self.cluster.centroid = new_centroid
        self.assertTrue(np.array_equal(self.cluster.centroid, new_centroid))


class TestRoutines(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.cluster = Cluster(np.array([3, 4, 5, 6]), self.sequence)
        self.routines = Routines(self.cluster)

    def test_add_routine(self):
        new_cluster = Cluster(np.array([7, 8, 9, 10]), self.sequence)
        self.routines.add_routine(new_cluster)
        self.assertEqual(len(self.routines), 2)

    def test_drop_indexes(self):
        self.routines.add_routine(self.cluster)
        self.routines = self.routines.drop_indexes([0])
        self.assertEqual(len(self.routines), 1)

    def test_get_routines(self):
        self.assertEqual(self.routines.get_routines(), [self.cluster])

    def test_to_collection(self):
        collection = self.routines.to_collection()
        expected_collection = [{'centroid': np.array([3, 4, 5, 6]), 'instances': self.sequence.to_collection()}]
        self.assertEqual(np.array_equal(collection[0]["centroid"], expected_collection[0]["centroid"]), True)
        self.assertEqual(collection[0]["instances"], expected_collection[0]["instances"])


if __name__ == '__main__':
    unittest.main()
