import unittest
import sys
import datetime
import numpy as np
import pandas as pd

sys.path.append('..')

from src.structures import Subsequence, Sequence, Cluster, Routines
from src.DRFL import DRFL


class TestSubsequence(unittest.TestCase):
    """
    Test the Subsequence class

    The Subsequence class is a class that represents a subsequence of a time series. It has the following methods:

        * __init__: initializes the Subsequence object
        * Distance: returns the distance between two subsequences
        * Magnitude: returns the magnitude of the subsequence
        * __len__: returns the length of the subsequence
        * __getitem__: returns the element at the index i
        * __eq__: returns True if the two subsequences are equal, False otherwise
    """

    def setUp(self):
        """
        Set up the Subsequence object and the subsequences for the tests

        The date is 2021-1-1

        The subsequences have the following parameters:

                * Subsequence 1: [1, 2, 3, 4], date: 2021-1-1, starting_point: 0
                * Subsequence 2: [5, 6, 7, 8], date: 2021-1-2, starting_point: 4
                * Subsequence 3: [9, 10, 11, 12], date: 2021-1-3, starting_point: 8
                * Subsequence 4: [13, 14, 15, 16], date: 2021-1-4, starting_point: 12
        """

        self.date = datetime.date(2021, 1, 1)
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.subsequence3 = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)
        self.subsequence4 = Subsequence(np.array([13, 14, 15, 16]), datetime.date(2021, 1, 4), 12)

    def test_init(self):
        """
        Test the __init__ method of the Subsequence class

        The method should initialize the Subsequence object with the following parameters:

                * instance: np.array([1, 2, 3, 4])
                * date: 2021-1-1
                * starting_point: 0
        """

        instance = np.array([1, 2, 3, 4])
        starting_point = 0
        self.assertEqual(np.array_equal(self.subsequence1.get_instance(), instance), True)
        self.assertEqual(self.subsequence1.get_date(), self.date)
        self.assertEqual(self.subsequence1.get_starting_point(), starting_point)

    def test_Distance(self):
        """
        Test the Distance method of the Subsequence class

        The method should return the distance between two subsequences

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * 4
        """

        self.assertEqual(self.subsequence1.Distance(self.subsequence2), 4)

    def test_Magnitude(self):
        """
        Test the Magnitude method of the Subsequence class

        The method should return the magnitude of the subsequence

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4
                * Subsequence 3: instances=[9, 10, 11, 12], date=2021-1-3, starting_point=8
                * Subsequence 4: instances=[13, 14, 15, 16], date=2021-1-4, starting_point=12

        The expected output is:
                * 4 for Subsequence 1
                * 8 for Subsequence 2
                * 12 for Subsequence 3
                * 16 for Subsequence 4
        """

        self.assertEqual(self.subsequence1.Magnitude(), 4)
        self.assertEqual(self.subsequence2.Magnitude(), 8)
        self.assertEqual(self.subsequence3.Magnitude(), 12)
        self.assertEqual(self.subsequence4.Magnitude(), 16)

    def test__len__(self):
        """
        Test the __len__ method of the Subsequence class

        The method should return the length of the subsequence

        The subsequence tested is:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * 4
        """
        self.assertEqual(len(self.subsequence1), 4)

    def test__getitem__(self):
        """
        Test the __getitem__ method of the Subsequence class

        The method should return the element at the index i

        The subsequence tested is:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * 1 for index 0
                * 2 for index 1
                * 3 for index 2
                * 4 for index 3
        """

        self.assertEqual(self.subsequence1[0], 1)
        self.assertEqual(self.subsequence1[1], 2)
        self.assertEqual(self.subsequence1[2], 3)
        self.assertEqual(self.subsequence1[3], 4)

    def test__eq__(self):
        """
        Test the __eq__ method of the Subsequence class

        The method should return True if the two subsequences are equal, False otherwise

        The other subsequences are:
                * first: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * second: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=1
                * third: instances=[1, 2, 5, 4], date=2021-1-1, starting_point=0
                * fourth: instances=[1, 2, 3, 4], date=2024-1-1, starting_point=0

        The expected subsequence is:
                * Subsequence1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * True for the first subsequence
                * False for the second subsequence
                * False for the third subsequence
                * False for the fourth subsequence
        """
        # CASE 1: True
        first = Subsequence(np.array([1, 2, 3, 4]), self.date, 0)
        self.assertEqual(self.subsequence1 == first, True)

        # CASE 2: False
        second = Subsequence(np.array([1, 2, 3, 4]), self.date, 1)
        self.assertEqual(self.subsequence1 == second, False)

        # CASE 3: False
        third = Subsequence(np.array([1, 2, 5, 4]), self.date, 0)
        self.assertEqual(self.subsequence1 == third, False)

        # CASE 4: False
        fourth = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2024, 1, 1), 0)
        self.assertEqual(self.subsequence1 == fourth, False)


class TestSequence(unittest.TestCase):
    """
    Test the Sequence class

    The Sequence class is a class that represents a sequence of subsequences. It has the following methods:

            * __init__: initializes the Sequence object
            * add_sequence: adds a subsequence to the sequence
            * get_by_starting_point: returns the subsequence with the starting point t
            * get_starting_points: returns the starting points of the subsequences
            * get_dates: returns the dates of the subsequences
            * get_subsequences: returns the subsequences
    """

    def setUp(self):
        """
        Set up the Sequence object and the subsequences for the tests

        The subsequences have the following parameters:
                    * Subsequence 1: [1, 2, 3, 4], date: 2021-1-1, starting_point: 0
                    * Subsequence 2: [5, 6, 7, 8], date: 2021-1-2, starting_point: 4
        """

        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)

    def test_add_sequence(self):
        """
        Test the add_sequence method of the Sequence class

        The method should add a subsequence to the sequence

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * 1 for the length of the sequence after adding the first subsequence
                * 2 for the length of the sequence after adding the second subsequence
        """

        self.sequence.add_sequence(self.subsequence1)
        self.assertEqual(len(self.sequence), 1)

        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(len(self.sequence), 2)

    def test_get_by_starting_point(self):
        """
        Test the get_by_starting_point method of the Sequence class

        The method should return the subsequence with the starting point t

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * Subsequence 1 for starting point 0
                * Subsequence 2 for starting point 4
                * None for starting point 8
        """

        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_by_starting_point(0), self.subsequence1)
        self.assertEqual(self.sequence.get_by_starting_point(4), self.subsequence2)
        self.assertIsNone(self.sequence.get_by_starting_point(8))

    def test_get_starting_points(self):
        """
        Test the get_starting_points method of the Sequence class

        The method should return the starting points of the subsequences

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * [0, 4]
        """

        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_starting_points(), [0, 4])

    def test_get_dates(self):
        """
        Test the get_dates method of the Sequence class

        The method should return the dates of the subsequences

        The subsequences are:

                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * [2021-1-1, 2021-1-2]
        """
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertEqual(self.sequence.get_dates(), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)])

    def test_get_subsequences(self):
        """
        Test the get_subsequences method of the Sequence class

        The method should return the subsequences

        The subsequences are:
                * Subsequence 1: instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * Subsequence 2: instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4

        The expected output is:
                * [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        """

        expected_output = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.assertTrue(np.array_equal(self.sequence.get_subsequences(), expected_output))


class TestCluster(unittest.TestCase):
    """
    Test the Cluster class

    The Cluster class is a class that represents a cluster of subsequences. It has the following methods:

        * __init__: initializes the Cluster object
        * add_instance: adds a subsequence to the cluster
        * get_sequences: returns the sequences of the cluster
        * update_centroid: updates the centroid of the cluster
        * get_starting_points: returns the starting points of the subsequences
        * get_dates: returns the dates of the subsequences
        * centroid_getter: returns the centroid of the cluster
        * centroid_setter: sets the centroid of the cluster


    """

    def setUp(self):
        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.cluster = Cluster(centroid=np.array([10, 10, 10, 10]), instances=self.sequence)

    def test__eq__(self):
        """
        Test the __eq__ method of the Cluster class

        The method should return True if the two clusters are equal, False otherwise

        The other clusters are:
                * first: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * second: centroid=[10, 10, 10, 10], instances=[5, 6, 7, 8], date=2021-1-2, starting_point=4
                * third: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0
                * fourth: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected cluster is:
                * Cluster: centroid=[10, 10, 10, 10], instances=[1, 2, 3, 4], date=2021-1-1, starting_point=0

        The expected output is:
                * True for the first cluster
                * False for the second cluster
                * True for the third cluster
                * True for the fourth cluster
        """
        # CASE 1: True
        first = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        self.assertTrue(self.cluster == first)

        # CASE 2: False
        second = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        second.add_instance(self.subsequence2)
        self.assertFalse(self.cluster == second)

        # CASE 3: True
        third = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        self.assertTrue(self.cluster == third)

        # CASE 4: True
        fourth = Cluster(np.array([10, 10, 10, 10]), self.sequence)
        self.assertTrue(self.cluster == fourth)

    def test_add_instance(self):
        """
        Test the add_instance method of the Cluster class

        The method should add a subsequence to the cluster

        The instance of the cluster is formed by the following Sequence:
            Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]
            )

        The new subsequence to add is:
                * new subsequence: instances=[9, 10, 11, 12], date=2021-1-3, starting_point=8

        The expected output is:
                * 3 for the length of the cluster after adding the new subsequence
        """

        new_subsequence = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 8)
        self.cluster.add_instance(new_subsequence)
        self.assertEqual(len(self.cluster.get_sequences()), 3)

    def test_get_sequences(self):
        """

        Test the get_sequences method of the Cluster class

        The method should return the sequences of the cluster

        The instance of the cluster is formed by the following Sequence:

            Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]
            )

        The expected output is:
                * [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        """
        subsequences_cluster = self.cluster.get_sequences().get_subsequences()
        expected_output = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]

        self.assertTrue(np.array_equal(subsequences_cluster, expected_output))

    def test_update_centroid(self):
        """
        Test the update_centroid method of the Cluster class

        The method should update the centroid of the cluster

        The instance of the cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The actual centroid is:
        [10, 10, 10, 10]

        The expected centroid after executing the update_centroid method is:
        [3.0, 4.0, 5.0, 6.0]
        """

        self.cluster.update_centroid()
        self.assertTrue(np.array_equal(self.cluster.centroid, np.array([3.0, 4.0, 5.0, 6.0])))

    def test_get_starting_points(self):
        """
        Test the get_starting_points method of the Cluster class

        The method should return the starting points of the subsequences

        The instance of the cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The expected output is:
            * [0, 4]

        """

        self.assertEqual(self.cluster.get_starting_points(), [0, 4])

    def test_get_dates(self):
        """
        Test the get_dates method of the Cluster class

        The method should return the dates of the subsequences

        The cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The expected output is:
            * [2021-1-1, 2021-1-2]
        """

        self.assertEqual(self.cluster.get_dates(), [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)])

    def test_centroid_getter(self):
        """
        Test the centroid_getter method of the Cluster class

        The method should return the centroid of the cluster

        The cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The expected output is:
            * [10, 10, 10, 10]
        """

        self.assertTrue(np.array_equal(self.cluster.centroid, np.array([10, 10, 10, 10])))

    def test_centroid_setter(self):
        """
        Test the centroid_setter method of the Cluster class

        The method should set the centroid of the cluster

        The cluster is formed by the following Sequence:

        Cluster(
            centroid=[10, 10, 10, 10],

            instances= Sequence(
                list_sequences=[
                    Subsequence(instance=[1, 2, 3, 4], date=2021-1-1, starting_point=0),

                    Subsequence(instance=[5, 6, 7, 8], date=2021-1-2, starting_point=4)
                ]))

        The new centroid is:
        [1, 2, 3, 4]

        The expected centroid set is:
        [1, 2, 3, 4]
        """

        new_centroid = np.array([1, 2, 3, 4])
        self.cluster.centroid = new_centroid
        self.assertTrue(np.array_equal(self.cluster.centroid, new_centroid))


class TestRoutines(unittest.TestCase):
    """
    Test the Routines class

    The Routines class is a class that represents a set of clusters. It has the following methods:

        * __init__: initializes the Routines object
        * add_routine: adds a cluster to the routines
        * drop_indexes: drops the clusters with the indexes in the list
        * get_routines: returns the clusters
        * to_collection: returns the clusters in a collection

    """

    def setUp(self):
        """
        Set up the Routines object and the clusters for the tests

        The Routine is formed by the following cluster:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])
        """

        self.sequence = Sequence()
        self.subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        self.subsequence2 = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
        self.sequence.add_sequence(self.subsequence1)
        self.sequence.add_sequence(self.subsequence2)
        self.cluster = Cluster(np.array([3, 4, 5, 6]), self.sequence)
        self.routines = Routines(self.cluster)

    def test_add_routine(self):
        """
        Test the add_routine method of the Routines class

        The method should add a cluster to the routines

        The cluster to add is:

        Cluster(
            - centroid = [7, 8, 9, 10],
            - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
            - starting_points = [0, 4]
            - dates = [2021-1-1, 2021-1-2]
        )

        The expected length of the routines after adding the cluster is:
            * 2
        """

        new_cluster = Cluster(np.array([7, 8, 9, 10]), self.sequence)
        self.routines.add_routine(new_cluster)
        self.assertEqual(len(self.routines), 2)

    def test_drop_indexes(self):
        """
        Test the drop_indexes method of the Routines class

        The method should return a new routine instance without the clusters with the indexes in the list

        The routines are:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [7, 8, 9, 10],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                ),
                Cluster(
                    - centroid = [7, 8, 9, 10],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])

        The actual length of the routines is:
                * 2

        The method is expected to drop the first cluster (index 0)

        The expected length of the routines after dropping the cluster is:
            * 1
        """

        self.routines.add_routine(self.cluster)
        self.routines = self.routines.drop_indexes([0])
        self.assertEqual(len(self.routines), 1)

    def test_get_routines(self):
        """
        Test the get_routines method of the Routines class

        The method should return a list of clusters

        The routines are:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])

        The expected output is:

        [Cluster(
            - centroid = [3, 4, 5, 6],
            - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
            - starting_points = [0, 4]
            - dates = [2021-1-1, 2021-1-2]
        )]
        """

        self.assertEqual(self.routines.get_routines(), [self.cluster])

    def test_to_collection(self):
        """
        Test the to_collection method of the Routines class

        The method should return the clusters in a collection

        The routines are:

        Routines(
            list_routines=[
                Cluster(
                    - centroid = [3, 4, 5, 6],
                    - instances = [[1, 2, 3, 4], [5, 6, 7, 8]]
                    - starting_points = [0, 4]
                    - dates = [2021-1-1, 2021-1-2]
                )])

        The expected output is:

        >>> [{'centroid': np.array([3, 4, 5, 6]),
        >>> 'instances': [
        >>>  ...  {'instance': [1, 2, 3, 4], 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
        >>>  ...  {'instance': [5, 6, 7, 8], 'date': datetime.date(2021, 1, 2), 'starting_point': 4}
        >>> ] }]
        """
        collection = self.routines.to_collection()
        expected_collection = [{'centroid': np.array([3, 4, 5, 6]), 'instances': self.sequence.get_subsequences()}]
        self.assertEqual(np.array_equal(collection[0]["centroid"], expected_collection[0]["centroid"]), True)
        self.assertTrue(
            np.array_equal(collection[0]["instances"][0]["instance"], expected_collection[0]["instances"][0]))


class TestDRFL(unittest.TestCase):
    """
    Test the DRFL class

    The DRFL class is a class that implements the DRFL algorithm. The DRFL algorithm is a clustering algorithm that

    The DRFL class has the following methods:
        * __minimum_distance_index: returns the index of the minimum distance in the list or array
        * __extract_subsequence: extracts a subsequence from the time series
        * __IsMatch: returns True if the distance between the subsequences is less than epsilon, False otherwise
        * __SubGroup: returns a Routines object with the clusters obtained from the time series
    """

    def setUp(self):
        """
        Set up the DRFL object and the time series for the tests

        The parameters of the DRFL object are:
            * m = 3
            * G = 4
            * R = 2
            * C = 3
            * epsilon = 1

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]
        """
        self.m = 3
        self.G = 4
        self.R = 2
        self.C = 3
        self.epsilon = 1
        self.time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
        self.time_series.index = pd.date_range(start="2024-01-01", periods=len(self.time_series))
        self.drfl = DRFL(m=3, G=4, R=2, C=3, epsilon=1)
        self.drfl_fitted = DRFL(m=3, G=4, R=2, C=3, epsilon=1)
        self.drfl_fitted.fit(self.time_series)

    def test__minimum_distance_index(self):
        """
        Test the __minimum_distance_index method of the DRFL class

        The method should return the index of the minimum distance in the list or array

        The list and the array have the following values:
        [1, 2, 0, 4]

        The expected output is:
        * 2

        There should be an expected type error for the following cases:
            * distances is not an index or an array
        """
        # Case 1: list as input
        distances = [1, 2, 0, 4]
        self.assertEqual(self.drfl._DRFL__minimum_distance_index(distances), 2)

        # Case 2: np.array as input
        distances = np.array([1, 2, 0, 4])
        self.assertEqual(self.drfl._DRFL__minimum_distance_index(distances), 2)

        # Case 3: Not an index or an array
        distances = "Not an index"
        with self.assertRaises(TypeError):
            self.drfl._DRFL__minimum_distance_index(distances)

    def test__extract_subsequence(self):
        """
        Test the __extract_subsequence method of the DRFL class

        The method should extract a subsequence from the time series

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]

        The expected output is:
            * Subsequence: [1, 3, 6], Date: 2024-1-1, Starting Point: 0

        There should be an expected type error for the following case:
            * t is not an integer

        There should be an expected value error for the following case:
            * t is out of the range of the time series
        """

        # Case 1: extracts correctly the first subsequence
        self.drfl._DRFL__extract_subsequence(self.time_series, 0)
        sequence = self.drfl._DRFL__sequence.get_by_starting_point(0)
        self.assertEqual(sequence.get_instance().tolist(), [1, 3, 6])
        self.assertEqual(sequence.get_date(), datetime.date(2024, 1, 1))
        self.assertEqual(sequence.get_starting_point(), 0)

        # Case 2: check if t is an integer
        with self.assertRaises(TypeError):
            self.drfl._DRFL__extract_subsequence(self.time_series, 0.5)
            self.drfl._DRFL__extract_subsequence(self.time_series, "not a integer")

        # Case 3: check if t is in the range of the time series
        with self.assertRaises(ValueError):
            self.drfl._DRFL__extract_subsequence(self.time_series, -1)
            self.drfl._DRFL__extract_subsequence(self.time_series, 100)

    def test__IsMatch(self):
        """
        Test the __IsMatch method of the DRFL class

        The method should return True if the distance between the subsequences is less than epsilon, False otherwise

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]

        The parameters of the DRFL object are:
              * R = 2

        The subsequences are:
            * Subsequence1: [1, 2, 3], Date: 2024-1-1, Starting Point: 0
            * Subsequence2: [3, 4, 5], Date: 2024-1-2, Starting Point: 0
            * Subsequence3: [3, 4, 6], Date: 2024-1-2, Starting Point: 0

        The expected output is:
            * True for Subsequence1 and Subsequence2
            * False for Subsequence1 and Subsequence3

        There should be an expected type error for the following cases:
            * S1 is not a Subsequence
            * S2 is not a Subsequence or an array
        """

        subsequence1 = Subsequence(np.array([1, 2, 3]), datetime.date(2024, 1, 1), 0)
        subsequence2 = Subsequence(np.array([3, 4, 5]), datetime.date(2024, 1, 2), 0)
        subsequence3 = Subsequence(np.array([3, 4, 6]), datetime.date(2024, 1, 2), 0)

        # Case 1: S1 Subsequence; S2: Subsequence
        self.assertTrue(self.drfl._DRFL__IsMatch(S1=subsequence1, S2=subsequence2, R=self.R))
        self.assertFalse(self.drfl._DRFL__IsMatch(S1=subsequence1, S2=subsequence3, R=self.R))

        # Case 2: S1 Subsequence; S2: Array
        self.assertTrue(self.drfl._DRFL__IsMatch(S1=subsequence1, S2=np.array([1, 2, 3]), R=self.R))
        self.assertFalse(self.drfl._DRFL__IsMatch(S1=subsequence1, S2=np.array([1, 2, 6]), R=self.R))

        # Case 3: S1 other type than Subsequence; S2: Subsequence
        with self.assertRaises(TypeError):
            self.drfl._DRFL__IsMatch(S1=np.array([1, 2, 3]), S2=subsequence1, R=self.R)
            self.drfl._DRFL__IsMatch(S1="Not a Subsequence", S2=subsequence1, R=self.R)

        # Case 4: S1 Subsequence; S2: other type than array an instance
        with self.assertRaises(TypeError):
            self.drfl._DRFL__IsMatch(S1=subsequence1, S2="Not an array", R=self.R)
            self.drfl._DRFL__IsMatch(S1=subsequence1, S2=[1, 2, 3], R=self.R)

    def test__SubGroup(self):
        """
        Test the __SubGroup method of the DRFL class

        The method should return a Routines object with the clusters obtained from the time series

        The time series is:
        [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]

        The parameters of the DRFL object are:
           * m = 3
           * G = 4
           * R = 2
           * C = 3
           * epsilon = 1

        The expected output is:

        Cluster 1:
            Centroid: [4/3, 3.0, 6.0]

            Instances:
                - Subsequence: [1, 3, 6], Date: 2024-1-1, Starting Point: 0
                - Subsequence: [2, 3, 6], Date: 2024-1-7, Starting Point: 6
                - Subsequence: [1, 3, 6], Date: 2024-1-12, Starting Point: 11

        Cluster 2:
            Centroid: [3.0, 6.0, 4.0]

            Instances:
                - Subsequence: [3, 6, 4], Date: 2024-1-2, Starting Point: 1
                - Subsequence: [3, 6, 4], Date: 2024-1-8, Starting Point: 7
                - Subsequence: [3, 6, 4], Date: 2024-1-13, Starting Point: 12

        Cluster 3:
            Centroid: [5.5, 3.5, 1.25]

            Instances:
                - Subsequence: [6, 4, 2], Date: 2024-1-3, Starting Point: 2
                - Subsequence: [4, 2, 1], Date: 2024-1-4, Starting Point: 3
                - Subsequence: [6, 4, 1], Date: 2024-1-9, Starting Point: 8
                - Subsequence: [6, 4, 1], Date: 2024-1-14, Starting Point: 13
        """

        # Expected instances for the first cluster
        expected_instances_centroid1 = Sequence(Subsequence(np.array([1, 3, 6]), pd.to_datetime("2024-1-1"), 0))
        expected_instances_centroid1.add_sequence(Subsequence(np.array([2, 3, 6]), pd.to_datetime("2024-1-7"), 6))
        expected_instances_centroid1.add_sequence(Subsequence(np.array([1, 3, 6]), pd.to_datetime("2024, 1, 12"), 11))

        # Expected instances for the second cluster
        expected_instances_centroid2 = Sequence(Subsequence(np.array([3, 6, 4]), pd.to_datetime("2024-1-2"), 1))
        expected_instances_centroid2.add_sequence(Subsequence(np.array([3, 6, 4]), pd.to_datetime("2024-1-8"), 7))
        expected_instances_centroid2.add_sequence(Subsequence(np.array([3, 6, 4]), pd.to_datetime("2024-1-13"), 12))

        # Expected instances for the third cluster
        expected_instances_centroid3 = Sequence(Subsequence(np.array([6, 4, 2]), pd.to_datetime("2024-1-3"), 2))
        expected_instances_centroid3.add_sequence(Subsequence(np.array([4, 2, 1]), pd.to_datetime("2024-1-4"), 3))
        expected_instances_centroid3.add_sequence(Subsequence(np.array([6, 4, 1]), pd.to_datetime("2024-1-9"), 8))
        expected_instances_centroid3.add_sequence(Subsequence(np.array([6, 4, 1]), pd.to_datetime("2024-1-14"), 13))

        expected_routine = Routines(Cluster(np.array([4 / 3, 3.0, 6.0]), expected_instances_centroid1))
        expected_routine.add_routine(Cluster(np.array([3.0, 6.0, 4.0]), expected_instances_centroid2))
        expected_routine.add_routine(Cluster(np.array([5.5, 3.5, 1.25]), expected_instances_centroid3))

        # Check if the routine is the expected
        routines_obtained_1 = self.drfl_fitted._DRFL__SubGroup(self.R, self.C, self.G)
        routines_obtained_2 = self.drfl_fitted._DRFL__SubGroup(R=3, C=2, G=2)

        self.assertEqual(routines_obtained_1, expected_routine)
        self.assertNotEquals(routines_obtained_2, expected_routine)


if __name__ == '__main__':
    unittest.main()
