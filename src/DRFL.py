"""
Discovering Routines of Fixed Length.

This script allows to discover routines of fixed length in a time series. The algorithm is based on the paper "An incremental algorithm for discovering routine behaviors from smart meter data" by Jin Wang, Rachel Cardell-Oliver and Wei Liu.

The algorithm is based on the following steps:

    * Extract subsequences of fixed length from the time series.
    * Group the subsequences into clusters based on their magnitude and maximum absolute distance.
    * Filter the clusters based on their frequency.
    * Test and handle overlapping clusters.

The algorithm is implemented in the class DRFL, which has the following methods and parameters:

The parameters:
    * m: Length of each secuence
    * R: Distance threshold
    * C: Frequency threshold
    * G: Magnitude threshold
    * epsilon: Overlap Parameter

Public methods:
    * fit: Fit the time series to the algorithm.
         Parameters:
            - time_series: Temporal data.
    * show_results: Show the results of the algorithm.
    * get_results: Returns the object Routines, with the discovered routines.
    * plot_results: Plot the results of the algorithm.
        Parameters:
            - title_fontsize: `Optional[int]`. Size of the title plot.
            - ticks_fontsize: `Optional[int]`. Size of the ticks.
            - labels_fontsize: `Optional[int]`. Size of the labels.
            - figsize: `Optional[tuple[int, int]]`. Size of the figure.
            - xlim: `Optional[tuple[int, int]]`. Limit of the x axis with starting points.
            - save_dir: `Optional[str]`. Directory to save the plot.

"""
import sys
import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

from itertools import product

sys.path.append("../")

from src.structures import Subsequence, Sequence, Cluster, Routines


class DRFL:
    """
    Discovering Routines of Fixed Length.

    This class allows to discover routines of fixed length in a time series. The algorithm is based on the paper "An incremental algorithm for discovering routine behaviors from smart meter data" by Jin Wang, Rachel Cardell-Oliver and Wei Liu.

    The algorithm is based on the following steps:

            * Extract subsequences of fixed length from the time series.
            * Group the subsequences into clusters based on their magnitude and maximum absolute distance.
            * Filter the clusters based on their frequency.
            * Test and handle overlapping clusters.

    Parameters:
        * m: Length of each secuence
        * R: Distance threshold
        * C: Frequency threshold
        * G: Magnitude threshold
        * epsilon: Overlap Parameter

    Methods
    _______
        * fit: Fit the time series to the algorithm.
        * show_results: Show the results of the algorithm.
        * get_results: Returns the object Routines, with the discovered routines.
        * plot_results: Plot the results of the algorithm.
        * estimate_distance: Estimate the customized distance from the obtained centroids and the target centroids.

    Examples:
    --------

        >>> from DRFL import DRFL
        >>> import pandas as pd

        >>> time_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
        >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
        >>> drfl.fit(time_series)
        >>> drfl.show_results()
        >>> drfl.plot_results()

    """

    def __init__(self, m: int, R: Union[float, int], C: int, G: Union[float, int], epsilon: float):
        """
        Initialize the DRFL algorithm.

        Parameters:
            * m: `int`. Length of each subsequence.
            * R: `float` or `int`. Distance threshold.
            * C: `int`. Frequency threshold.
            * G: `float` or `int`. Magnitude threshold.
            * epsilon: `float`. Overlap parameter.

        Examples:
            >>> from DRFL import DRFL
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
        """

        self.m: int = m
        self.R: int | float = R
        self.C: int = C
        self.G: int | float = G
        self.epsilon: float = epsilon
        self.__routines: Routines = Routines()
        self.__sequence: Sequence = Sequence()

        self.__already_fitted: bool = False
        self.time_series: pd.Series = None

    @staticmethod
    def __check_type_time_series(time_series: pd.Series) -> None:
        """
        Check the type of the time series.

        Parameters:
            * time_series: `pd.Series`. Temporal data.

        Raises:
            TypeError: If the time series is not a `pandas Series` with `DatetimeIndex` at the index.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Examples:
            >>> from DRFL import DRFL
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 2, 3, 4, 5])
            >>> DRFL.__check_type_time_series(time_series)
            TypeError: time_series must be a pandas Series with a DatetimeIndex
        """

        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")

        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise TypeError("time_series index must be a pandas DatetimeIndex")

    @staticmethod
    def __minimum_distance_index(distances: Union[np.ndarray, list]) -> int:
        """
        Get the index of the minimum distance in a list of distances.

        Parameter:
            distances: `np.array` or `list`. List of distances.

        Returns:
             `int`. Index of the minimum distance.

        Raises:
            TypeError: If the distances are not a list or a numpy array.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Examples:
            >>> from DRFL import DRFL
            >>> distances = [1, 2, 3, 4, 5]
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__minimum_distance_index(distances)
            0
        """
        # Check if the distances are a list
        if not isinstance(distances, list) and not isinstance(distances, np.ndarray):
            raise TypeError("distances must be a list or a numpy array")

        return int(np.argmin(distances))

    @staticmethod
    def __IsMatch(S1: Subsequence, S2: Union[np.ndarray, Subsequence], R: int | float) -> bool:
        """
        Check if two subsequences match by checking if the distance between them is lower than the threshold distance parameter R.

        Parameters:
            * S1: `Subsequence`. The first subsequence.
            * S2: `np.array` or `Subsequence`. The second subsequence.
            * R: `int` or `float`. The threshold distance parameter.

        Returns:
            `bool`. `True` if the distance between the subsequences is lower than the threshold distance parameter R, `False` otherwise.

        Raises:
            TypeError: If S1 is not an instance of `Subsequence` or S2 is not an instance of `Subsequence` or `np.ndarray`.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__IsMatch(S1, S2, 2)
            True

            >>> S3 = Subsequence(instance=np.array([1, 2, 6]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> drfl.__IsMatch(S1, S3, 2)
            False
        """

        # Check if S1 is an instance of Subsequence
        if not isinstance(S1, Subsequence):
            raise TypeError("S1 must be instance of Subsequence")

        # Check if S2 is an instance of Subsequence or np.ndarray
        if isinstance(S2, Subsequence) or isinstance(S2, np.ndarray):
            return S1.Distance(S2) <= R

        raise TypeError("S2 must be instance of Subsequence or np.ndarray")

    @staticmethod
    def __IsOverlap(S_i: Subsequence, S_j: Subsequence):
        """
        Check if two subsequences overlap by applying the following inequality from the paper:

        (i + p) > j or (j + q) > i

        Where:
            * i: Starting point of the first subsequence.
            * j: Starting point of the second subsequence.
            * p: Length of the first subsequence.
            * q: Length of the second subsequence.

        Parameters:
            * S_i: `Subsequence`. The first subsequence with starting point i.
            * S_j: `Subsequence`. The second subsequence with starting point j.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Returns:
             `True` if they overlap, `False` otherwise.

        Raises:
            TypeError: If S_i or S_j are not instances of Subsequence.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__IsOverlap(S1, S2)
            True

            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=4)
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__IsOverlap(S1, S2)
            False
        """

        # Check if S_i and S_j are instances of Subsequence
        if not isinstance(S_i, Subsequence) or not isinstance(S_j, Subsequence):
            raise TypeError("S_i and S_j must be instances of Subsequence")

        start_i, p = S_i.get_starting_point(), len(S_i.get_instance())
        start_j, q = S_j.get_starting_point(), len(S_j.get_instance())
        return not ((start_i + p <= start_j) or (start_j + q <= start_i))

    @staticmethod
    def __inverse_gaussian_distance(N_target: int, N_estimated: int, sigma: float) -> float:
        """
        Compute the inverse gaussian distance between the target and estimated number of instances.

        It applies the following formula:

        1 - exp(-((N_target - N_estimated) ** 2) / sigma)

        This distance is ranged from 0 to 1, where 0 means that the target and estimated number of instances are equal and 1 means that they are different.
        Its purpose is to penalize the difference between the target and estimated number of routines in a smooth way.

        Parameters:
            * N_target: `int`. Target number of centroids.
            * N_estimated: `int`. Estimated number of centroids.
            * sigma: `float`. Standard deviation parameter for the inverse gaussian distance calculation. Lower values of sigma penalizes more the difference between the target and estimated number of centroids.

        Returns:
            `float`. The inverse gaussian distance between the target and estimated number of instances.

        Examples:
            >>> from DRFL import DRFL
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__inverse_gaussian_distance(N_target=3, N_estimated=3, sigma=2)
            0
        """

        return 1 - np.exp(-((N_target - N_estimated) ** 2) / sigma)

    @staticmethod
    def __matrix_of_distances(target_centroids: list[list], estimated_centroids: list[np.ndarray]) -> np.ndarray:
        """
        Compute the matrix of distances between the target and estimated centroids.
        The distance between the target and estimated centroids is the maximum absolute difference between them.

        Parameters:
            * target_centroids: `list[list]`. Target centroids.
            * estimated_centroids: `list[np.ndarray]`. Estimated centroids.

        Returns:
            `np.array`. Matrix of distances between the target and estimated centroids.

        Examples:
            >>> target_centroids = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            >>> estimated_centroids = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]
            >>> drfl.__matrix_of_distances(target_centroids, estimated_centroids)
            array([[0., 1., 2.],
                   [1., 0., 1.],
                   [2., 1., 0.]])
        """

        # Initialize the matrix of distances
        matrix = np.zeros((len(target_centroids), len(estimated_centroids)))

        # Compute the matrix of distances
        for i, target in enumerate(target_centroids):
            for j, estimated in enumerate(estimated_centroids):
                # Compute the distance between the target and estimated centroids
                matrix[i, j] = np.max(np.abs(np.array(target) - estimated))

        return matrix

    @staticmethod
    def __closest_centroids_distances(matrix_of_distances: np.ndarray) -> np.ndarray:
        """
        Compute the closest distances between the target and estimated centroids.

        Parameters:
            * matrix_of_distances: `np.ndarray`. Matrix of distances between the target and estimated centroids.

        Returns:
            `np.ndarray`. Closest distances between the target and estimated centroids.
        """

        return np.min(matrix_of_distances, axis=1)

    def __extract_subsequence(self, time_series: pd.Series, t: int) -> None:
        """
        Extract a subsequence from the time series and adds the subsequence to Sequence object.

        Parameters:
            * time_series: `pd.Series`. Temporal data.
            * t: `int`. Starting point of the subsequence.

        Raises:
            TypeError: If t is not an integer or time_series is not a pandas Series.
            ValueError: If the starting point of the subsequence is out of the time series range.

        Notes:
            This method cannot be accessed from outside

        Examples:
            >>> from DRFL import DRFL
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 2, 3, 4, 5])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__extract_subsequence(time_series, 0) # This property cannot be accessed from outside the class
            >>> print(drfl.__sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3]),
                        date=datetime.date(2024, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([2, 3, 4]),
                        date=datetime.date(2024, 1, 2),
                        starting_point=1
                    ),
                    Subsequence(
                        instance=np.array([3, 4, 5]),
                        date=datetime.date(2024, 1, 3),
                        starting_point=2
                    ),
                    Subsequence(
                        instance=np.array([4, 5, 6]),
                        date=datetime.date(2024, 1, 4),
                        starting_point=3
                    ),
                ]
            )
        """
        # Check if time_series is a pandas series
        self.__check_type_time_series(time_series)

        # Check if t is an integer
        if not isinstance(t, int):
            raise TypeError("t must be an integer")

        # Check if t is within the range of the time series
        if t + self.m > len(time_series) or t < 0:
            raise ValueError(f"The starting point of the subsequence is out of the time series range")

        window = time_series[t:t + self.m]  # Extract the time window

        subsequence = Subsequence(instance=window.values,
                                  date=time_series.index[t],
                                  starting_point=t)  # Get the subsequence from the time window

        self.__sequence.add_sequence(subsequence)  # Add the subsequence to the sequences

    def __NotTrivialMatch(self, subsequence: Subsequence, cluster: Cluster, start: int, R: int | float) -> bool:
        """
        Checks if a subsequence is not a trivial match with any of the instances from the cluster.

        This method returns False if there is not a match between the
        subsequence and the centroid.
        It also returns False if there is a match between the subsequence
        and any subsequence with a starting point between the start
        parameter and the starting point of the subsequence.
        Otherwise, it returns True.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Parameters:
            * subsequence: `Subsequence`. The subsequence to check.
            * cluster: `Cluster`. The cluster to check.
            * start: `int`. Starting point of the subsequence.
            * R: `int` or `float`. The threshold distance parameter.

        Returns:
            `bool`. `True` if the subsequence is not a trivial match with any of the instances from the cluster, `False` otherwise.

        Raises:
             TypeError: If subsequence is not an instance of `Subsequence` or cluster is not an instance of `Cluster`.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__NotTrivialMatch(S1, cluster, 0, 2)
            False
            >>> drfl.__NotTrivialMatch(S1, cluster, 1, 2)
            True
        """

        # Check if subsequence is an instance of Subsequence and cluster is an instance of Cluster
        if not isinstance(subsequence, Subsequence) or not isinstance(cluster, Cluster):
            raise TypeError("subsequence and cluster must be instances of Subsequence and Cluster respectively")

        # Check if the subsequence is not a trivial match with any of the instances from the cluster
        if not self.__IsMatch(S1=subsequence, S2=cluster.centroid, R=R):
            return False

        # Check if there is a match between the subsequence and any subsequence with a starting point
        # between the start parameter and the starting point of the subsequence
        for end in cluster.get_starting_points():
            for t in reversed(range(start + 1, end)):
                # If some subsequence is a trivial match with a subsequence from the referenced
                # starting point, it returns False
                if self.__IsMatch(S1=subsequence, S2=self.__sequence.get_by_starting_point(t), R=R):
                    return False

        return True

    def __SubGroup(self, R: float | int, C: int, G: float | int) -> Routines:
        """
        Group the subsequences into clusters based on their magnitude and maximum absolute distance.

        The steps that follow this algorithm are:
            * Create a new cluster with the first subsequence.
            * For each subsequence, check if it is not a trivial match with any of the instances within the cluster.
            * If it is not a trivial match, append new Sequence on the instances of the cluster.
            * If it is a trivial match, create a new cluster.
            * Filter the clusters by frequency.

        Notes:
            This method is private and cannot be accessed from outside the class.

        Parameters:
            * R: `float` or `int`. Distance threshold.
            * C: `int`. Frequency threshold.
            * G: `float` or `int`. Magnitude threshold.

        Returns:
            Routines. The clusters of subsequences.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1)
            >>> drfl.fit(time_series)
            >>> routines = drfl.__SubGroup()
            >>> print(routines)
            Routines(
            list_routines=[
                Cluster(
                    -Centroid: [1.33333333 3.         6.        ]
                    -Instances: [array([1, 3, 6]), array([2, 3, 6]), array([1, 3, 6])]
                    -Dates: [Timestamp('2024-01-01 00:00:00'), Timestamp('2024-01-07 00:00:00'), Timestamp('2024-01-12 00:00:00')]
                    -Starting Points: [0, 6, 11]
                ),
                Cluster(
                    -Centroid: [3. 6. 4.]
                    -Instances: [array([3, 6, 4]), array([3, 6, 4]), array([3, 6, 4])]
                    -Dates: [Timestamp('2024-01-02 00:00:00'), Timestamp('2024-01-08 00:00:00'), Timestamp('2024-01-13 00:00:00')]
                    -Starting Points: [1, 7, 12]
                ),
                Cluster(
                    -Centroid: [5.5  3.5  1.25]
                    -Instances: [array([6, 4, 2]), array([4, 2, 1]), array([6, 4, 1]), array([6, 4, 1])]
                    -Dates: [Timestamp('2024-01-03 00:00:00'), Timestamp('2024-01-04 00:00:00'), Timestamp('2024-01-09 00:00:00'), Timestamp('2024-01-14 00:00:00')]
                    -Starting Points: [2, 3, 8, 13]
                )]
            )
        """

        routines = Routines()

        # Iterate through all the subsequences
        for i in range(len(self.__sequence)):
            if self.__sequence[i].Magnitude() >= G:  # Check if the magnitude of the subsequence is greater than G
                if routines.is_empty():  # Initialize first cluster if its empty
                    # Create a cluster from the first subsequence
                    routines.add_routine(Cluster(centroid=self.__sequence[i].get_instance(),
                                                 instances=Sequence(subsequence=self.__sequence[i])))
                    continue  # Continue to the next iteration

                # Estimate all the distances between the subsequence and all the centroids of the clusters
                distances = [self.__sequence[i].Distance(routines[j].centroid) for j in range(len(routines))]

                # Get the index of the minimum distance to the centroid
                j_hat = self.__minimum_distance_index(distances)

                # Check if the subsequence is not a trivial match with any of the instances within the cluster
                # if self.__NotTrivialMatch(subsequence=self.sequence[i], cluster=routines[j_hat], start=i, R=R):
                if self.__IsMatch(S1=self.__sequence[i], S2=routines[j_hat].centroid, R=R):
                    routines[j_hat].add_instance(self.__sequence[i])  # Append new Sequence on the instances of Bm_j
                    routines[j_hat].update_centroid()  # Update center of the cluster

                else:
                    # create a new cluster//routine
                    new_cluster = Cluster(centroid=self.__sequence[i].get_instance(),
                                          instances=Sequence(subsequence=self.__sequence[i]))
                    routines.add_routine(new_cluster)  # Add the new cluster to the routines

        # Filter by frequency
        to_drop = [k for k in range(len(routines)) if len(routines[k]) < C]
        filtered_routines = routines.drop_indexes(to_drop)

        return filtered_routines

    def __OLTest(self, cluster1: Cluster, cluster2: Cluster, epsilon: float) -> tuple[bool, bool]:
        """
        Test and handle overlapping clusters by determining the significance of their overlap.

        Overlapping clusters are analyzed to decide if one, both, or none should be kept based on the overlap
        percentage and the clusters' characteristics. This determination is crucial for maintaining the
        quality and interpretability of the detected routines. The method employs a two-step process: first,
        it calculates the number of overlapping instances between the two clusters; then, based on the overlap
        percentage and the clusters' properties (e.g., size and magnitude), it decides which cluster(s) to retain.

        Parameters:
            * cluster1: `Cluster`. The first cluster involved in the overlap test.
            * cluster2: `Cluster`. The second cluster involved in the overlap test.
            * epsilon: `float`. A threshold parameter that defines the minimum percentage of overlap required for considering an overlap significant. Values range from 0 to 1, where a higher value means a stricter criterion for significance.

        Returns:
            * tuple[bool, bool]: A tuple containing two boolean values. The first value indicates whether
                                 cluster1 should be kept (True) or discarded (False). Similarly, the second
                                 value pertains to cluster2.


        Overview of the Method's Logic:
            * Calculate the number of instances in cluster1 that significantly overlap with any instance in cluster2.
            * determine the significance of the overlap based on the 'epsilon' parameter: if the number of overlaps exceeds 'epsilon' times the smaller cluster's size, the overlap is considered significant.
            * In case of significant overlap, compare the clusters based on their size and the cumulative magnitude of their instances. The cluster with either a larger size or a greater cumulative magnitude (in case of a size tie) is preferred.
            * Return a tuple indicating which clusters should be kept. If the overlap is not significant, both clusters may be retained.

        Note:
            * This method relies on private helper methods to calculate overlaps and compare cluster properties.
            * The method does not modify the clusters directly but provides guidance on which clusters to keep or discard.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> import pandas as pd
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__OLTest(cluster1, cluster2, 0.5)
            (True, False)
        """

        N = 0  # Initialize counter for number of overlaps

        # Iterate through all instances in cluster1
        for S_i in cluster1.get_sequences():
            # Convert instance to Subsequence if needed for overlap checks
            for S_j in cluster2.get_sequences():
                # Check for overlap between S_i and S_j
                if self.__IsOverlap(S_i, S_j):
                    N += 1  # Increment overlap count
                    break  # Break after finding the first overlap for S_i

        # Calculate the minimum length of the clusters to determine significance of overlap
        min_len = min(len(cluster1), len(cluster2))

        # Determine if the overlap is significant based on epsilon and the minimum cluster size
        if N > epsilon * min_len:

            # Calculate cumulative magnitudes for each cluster to decide which to keep
            mag_cluster1 = cluster1.cumulative_magnitude()
            mag_cluster2 = cluster2.cumulative_magnitude()

            # Keep the cluster with either more instances or, in a tie, the greater magnitude
            if len(cluster1) > len(cluster2) or (len(cluster1) == len(cluster2) and mag_cluster1 > mag_cluster2):
                return True, False
            else:
                return False, True
        else:
            # If overlap is not significant, propose to keep both clusters
            return True, True

    def __obtain_keep_indices(self, epsilon: float) -> list[int]:
        """
        Obtain the indices of the clusters to keep based on the overlap test.

        Parameters:
            epsilon: `float`. A threshold parameter that defines the minimum percentage of overlap required for considering an overlap significant. Values range from 0 to 1, where a higher value means a stricter criterion for significance.

        Returns:
            `list[int]`. The indices of the clusters to keep.

        Raises:
             ValueError: If epsilon is not between 0 and 1.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> import pandas as pd
            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=4)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
            >>> drfl.__obtain_keep_indices(1)
            [0, 1]

            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.__obtain_keep_indices(0.5)
            [1]
        """

        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be between 0 and 1")

        # Prepare to test and handle overlapping clusters
        keep_indices = set(range(len(self.__routines)))  # Initially, assume all clusters are to be kept

        for i in range(len(self.__routines) - 1):
            for j in range(i + 1, len(self.__routines)):
                if i in keep_indices and j in keep_indices:  # Process only if both clusters are still marked to keep
                    keep_i, keep_j = self.__OLTest(self.__routines[i], self.__routines[j], epsilon)

                    # Update keep indices based on OLTest outcome
                    if not keep_i:
                        keep_indices.remove(i)
                    if not keep_j:
                        keep_indices.remove(j)

        return list(keep_indices)

    def fit(self, time_series: pd.Series) -> None:
        """
        Fits the time series data to the `DRFL` algorithm to discover routines.

        This method preprocesses the time series data, extracts subsequences, groups them into clusters, and finally filters and handles overlapping clusters to discover and refine routines.

        Parameters:
             time_series: `pd.Series`. The time series data to analyze. It should be a `pandas Series` object with a `DatetimeIndex`.

        Raises:
             TypeError: If the input time series is not a `pandas Series` or if its index is not a `DatetimeIndex`.

        Examples:
            >>> from DRFL import DRFL
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
            >>> drfl.fit(time_series)
            >>> print(drfl.routines)
        """

        self.__check_type_time_series(time_series)
        self.time_series = time_series

        # Set the already_fitted attribute to True
        self.__already_fitted = True

        for i in range(len(self.time_series) - self.m + 1):
            self.__extract_subsequence(self.time_series, i)

        # Group the subsequences into clusters based on their magnitude and
        # maximum absolute distance and filter the clusters based on their frequency
        self.__routines = self.__SubGroup(R=self.R, C=self.C, G=self.G)

        # Obtain the indices of the clusters to keep based on the overlap test
        keep_indices = self.__obtain_keep_indices(self.epsilon)

        # Filter self.routines to keep only those clusters marked for keeping
        if len(self.__routines) > 0:
            to_drop = [k for k in range(len(self.__routines)) if k not in keep_indices]
            self.__routines = self.__routines.drop_indexes(to_drop)

        if len(self.__routines) == 0:
            warnings.warn("No routines have been discovered", UserWarning)

    def estimate_distance(self, target_centroids: list[list], alpha: float, sigma: float) -> float:
        """
        Estimate the distance between the target centroids and the estimated centroids.
        The distance is a combination of the penalization
        of detecting a distinct number of routines and the normalized distance between the target and estimated centroids.
        Applies the following formula:

        alpha * penalization + (1 - alpha) * normalized_distance

        Where penalization is the inverse gaussian distance between the target and estimated number of instances,
        and normalized_distance is the sum of the closest distances between the target and estimated centroids
        divided by the total sum of the distances.

        The result is a distance value ranged from 0 to 1,
        where 0 means that the target and estimated centroids are equal and 1 means that they are different.

        Parameters:
            * target_centroids: `list[list]`. Target centroids.
            * alpha: `float`. Weight parameter to balance the penalization and normalized distance.
            * sigma: `float`. Standard deviation parameter for the inverse gaussian distance calculation.

        Returns:
            `float`. The distance between the target and estimated centroids ranged from 0 to 1.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            TypeError: If target_centroids is not a list of lists.
            ValueError: If alpha is not between 0 and 1 or sigma is not greater than 1.

        Examples:
            >>> from DRFL import DRFL
            >>> import numpy as np
            >>> target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> drfl = DRFL(m=3, R=1, C=3, G=4, epsilon=1)
            >>> drfl.fit(time_series)
            >>> dist = drfl.estimate_distance(target_centroids, alpha=0.5, sigma=3)
            >>> print(dist)
            0.0
        """

        # Check if the model has been fitted
        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Check if there are routines to compare
        if self.__routines.is_empty():
            warnings.warn("No routines have been discovered", UserWarning)
            return np.nan

        # Estimate the penalization of detecting a distinct number of routines
        N_target, N_estimated = len(target_centroids), len(self.__routines)

        # estimate the penalization of detecting a distinct number of routines
        penalization = self.__inverse_gaussian_distance(N_target, N_estimated, sigma)

        # calculate the matrix of distances between the target centroids and the estimated centroids
        matrix_of_distances = self.__matrix_of_distances(target_centroids, self.__routines.get_centroids())

        # calculate the closest distances between the target centroids and the estimated centroids
        closest_distances = self.__closest_centroids_distances(matrix_of_distances)

        # Normalization of the distances
        total_sum_matrix = np.sum(matrix_of_distances)
        total_sum_closest = np.sum(closest_distances)

        # Avoid division by zero
        if total_sum_matrix == 0:
            normalized_distance = 0
        else:
            normalized_distance = total_sum_closest / total_sum_matrix

        return alpha * penalization + (1 - alpha) * normalized_distance

    def show_results(self) -> None:
        """
        Displays the discovered routines after fitting the model to the time series data.

        This method prints out detailed information about each discovered routine, including the centroid of each cluster, the subsequence instances forming the routine, and the dates/times these routines occur.

        Note:
            This method should be called after the `fit` method to ensure that routines have been discovered and are ready to be displayed.
        """

        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        print("Routines detected: ", len(self.__routines))
        print("_" * 50)
        for i, b in enumerate(self.__routines):
            print(f"Centroid {i + 1}: {b.centroid}")
            print(f"Routine {i + 1}: {b.get_sequences().get_subsequences()}")
            print(f"Date {i + 1}: {b.get_dates()}")
            print(f"Starting Points {i + 1}: ", b.get_starting_points())
            print("\n", "-" * 50, "\n")

    def get_results(self) -> Routines:
        """
        Returns the discovered routines as a `Routines` object.

        After fitting the model to the time series data, this method can be used to retrieve the discovered routines, encapsulated within a `Routines` object, which contains all the clusters (each representing a routine) identified by the algorithm.

        Returns:
             `Routines`. The discovered routines as a `Routines` object.

        Note:
            The `Routines` object provides methods and properties to further explore and manipulate the discovered routines.
        """

        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        return self.__routines

    def plot_results(self, title_fontsize: int = 20,
                     xticks_fontsize: int = 20, yticks_fontsize: int = 20,
                     labels_fontsize: int = 20, figsize: tuple[int, int] = (30, 10),
                     text_fontsize: int = 20, linewidth_bars: int = 1.5,
                     xlim: Optional[tuple[int, int]] = None,
                     save_dir: Optional[str] = None) -> None:

        """
        This method uses matplotlib to plot the results of the algorithm. The plot shows the time series data with vertical dashed lines indicating the start of each discovered routine. The color of each routine is determined by the order in which they were discovered, and a legend is displayed to identify each routine.

        Parameters:
            * title_fontsize: `int` (default is 20). Size of the title plot.
            * xticks_fontsize: `int` (default is 20). Size of the xticks.
            * yticks_fontsize: `int (default is 20)`. Size of the yticks.
            * labels_fontsize: `int` (default is 20). Size of the labels.
            * figsize: `tuple[int, int]` (default is (30, 10)). Size of the figure.
            * text_fontsize: `int` (default is 20). Size of the text.
            * linewidth_bars: `int` (default is 1.5). Width of the bars in the plot.
            * xlim: `tuple[int, int]` (default is None). Limit of the x axis with starting points.
            * save_dir: `str` (default is None). Directory to save the plot.

        Notes:
           This method has to be executed after the fit method to ensure that routines have been discovered and are ready to be displayed.
        """

        # Check if the model has been fitted
        if not self.__already_fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Generate a color map for the routines
        base_colors = cm.rainbow(np.linspace(0, 1, len(self.__routines)))

        # Convert the time series data to a numpy array for easier manipulation
        ts = np.array(self.time_series)

        # Create a new figure with the specified size
        plt.figure(figsize=figsize)

        # Get the number of routines and the maximum value in the time series
        N_rows = len(self.__routines)
        maximum = max(ts)

        # if xlim is not provided, set the limits of the x-axis to the range of the time series
        xlim = xlim or (0, len(ts) - 1)

        # Get the starting points of each routine
        start_points = [cluster.get_starting_points() for cluster in self.__routines]

        # For each routine, create a subplot and plot the routine
        for row, routine in enumerate(start_points):
            plt.subplot(N_rows, 1, row + 1)

            # Initialize the color of each data point in the time series as gray
            colors = ["gray"] * len(ts)

            # Set the title and x-label of the subplot
            plt.title(f'Routine {row + 1}', fontsize=title_fontsize)
            plt.xlabel("Starting Points", fontsize=labels_fontsize)

            # For each starting point in the routine, plot a vertical line and change the color of the data points in the routine
            for sp in routine:
                if xlim[0] <= sp <= xlim[1]:
                    plt.axvline(x=sp, color=base_colors[row], linestyle="--")
                    for j in range(self.m):
                        if sp + j <= xlim[1]:
                            plt.text(sp + j - 0.05, self.time_series[sp + j] - 0.8, f"{ts[sp + j]}",
                                     fontsize=text_fontsize, backgroundcolor="white", color=base_colors[row])
                            colors[sp + j] = base_colors[row]

            # Plot the time series data as a bar plot
            plt.bar(x=np.arange(0, len(ts)), height=ts, color=colors, edgecolor="black", linewidth=linewidth_bars)

            # Set the ticks on the x-axis
            plt.xticks(ticks=np.arange(xlim[0], xlim[1] + 1),
                       labels=np.arange(xlim[0], xlim[1] + 1),
                       fontsize=xticks_fontsize)

            plt.yticks(fontsize=yticks_fontsize)

            # Plot a horizontal line at the magnitude threshold
            plt.axhline(y=self.G, color="red", linestyle="--")

            # Set the limits of the x-axis and y-axis
            plt.xlim(xlim[0] - 0.5, xlim[1] + 0.5)
            plt.ylim(0, maximum + 1)

            # Adjust the layout of the plot
            plt.tight_layout()

        # If a directory is provided, save the figure to the directory
        if save_dir:
            plt.savefig(save_dir)

        # Display the plot
        plt.show()


class ParallelSearchDRFL:
    """
    Class to perform a parallel search of the best parameters for the DRFL algorithm using multithreading.

    This class allows the user to search for the best parameters for the DRFL algorithm using a grid search approach.

    Parameters:
        * n_jobs: `int`. The number of parallel jobs to run.
        * alpha: `int | float`. Rate of penalization.
        * sigma: `int | float`. Sigma parameter for the variance in the inverse Gaussian distance.
        * param_grid: `dict`. Dictionary with parameters names (`m`, `R`, `C`, `G`, `epsilon`) as keys and lists of their values to try, representing the parameters from the DRFL algorithm.

    Attributes:
    -------
        * n_jobs: `int`. The number of parallel jobs to run.
        * alpha: `int | float`. Rate of penalization.
        * sigma: `int | float`. Sigma parameter for the variance in the inverse Gaussian distance.
        * param_grid: `dict`. Dictionary with parameters names (`m`, `R`, `C`, `G`, `epsilon`) as keys and lists of their values to try, representing the parameters from the DRFL algorithm.

    Methods:
    -------
        * fit(time_series: pd.Series, target_centroids: list[list]): Fit the DRFL algorithm to the time series data using multiple sets of parameters in parallel.
        * best_params(): Get the best parameters found during the search.
        * cv_results(): Get the results of the search.


    Examples:
    --------
        >>> from DRFL import ParallelSearchDRFL
        >>> import pandas as pd
        >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
        >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
        >>> param_grid = {
        ...     "m": [3],
        ...     "R": [1, 2, 3, 4, 5, 6],
        ...     "C": [1, 2, 3, 4, 5, 6],
        ...     "G": [1, 2, 3, 4, 5, 6],
        ...     "epsilon": [0.5, 1],
        ... }
        >>> search = ParallelSearchDRFL(n_jobs=2, alpha=0.5, sigma=3, param_grid=param_grid)
        >>> search.fit(time_series)
        >>> print(search.best_params())
        >>> {'m': 3, 'R': 1, 'C': 3, 'G': 4, 'epsilon': 1}
    """

    def __init__(self, n_jobs: int, alpha: Union[int, float], sigma: Union[int, float], param_grid: dict):
        """
        Initialize the ParallelDRFL object with a parameter grid for the DRFL algorithm and the number of jobs for parallel processing.

        Parameters:
            * n_jobs: `int`. The number of parallel jobs to run.
            * alpha: `int | float`. Rate of penalization.
            * sigma: `int | float`. Sigma parameter for the variance in the inverse Gaussian distance.
            * param_grid: `dict`. Dictionary with parameters names (`m`, `R`, `C`, `G`, `epsilon`) as keys and lists of their values to try, representing the parameters from the DRFL algorithm.

        Raises:
            TypeError: If the parameter grid is not a dictionary.
            ValueError: If the parameter grid is empty or if some parameter is not valid or has an invalid value.
        """
        # Check the validity of the parameters
        self.__check_validity_params(alpha, sigma, param_grid)

        # Set the attributes
        self.__n_jobs: int = n_jobs
        self.__alpha: Union[int, float] = alpha
        self.__sigma: Union[int, float] = sigma
        self.__param_grid: dict = param_grid

        # Initialize the results attribute and the fitted attribute
        self.__results: list[dict] = []
        self.__fitted: bool = False

    @staticmethod
    def __check_m_param(m: int):
        """
        Check the validity of the m parameter.

        Parameters:
            * m: `int`. The m parameter to check.

        Raises:
            TypeError: If m is not an integer.
            ValueError: If m is not greater than 1.
        """

        if not isinstance(m, int):
            raise TypeError("m must be an integer")

        if m < 2:
            raise ValueError("m must be greater or equal than 2")

    @staticmethod
    def __check_R_or_G_param(param: list[Union[int, float]]):
        """
        Check the validity of the R parameter.

        Parameters:
            * R: `list[int | float]`. The R parameter to check.

        Raises:
            TypeError: If R is not a list.
            ValueError: If R is empty or if some value in R is not greater than one.
        """

        if not isinstance(param, list):
            raise TypeError("R must be a list")

        if len(param) == 0:
            raise ValueError("R cannot be empty")

        for value in param:
            if value < 1:
                raise ValueError("R must be greater or equal than 1")

    @staticmethod
    def __check_C_param(C: list[int]):
        """
        Check the validity of the C parameter.

        Parameters:
            * C: `list[int]`. The C parameter to check.

        Raises:
            TypeError: If C is not a list.
            ValueError: If C is empty or if some value in C is not greater than one.
        """

        if not isinstance(C, list):
            raise TypeError("C must be a list")

        if len(C) == 0:
            raise ValueError("C cannot be empty")

        for value in C:
            if value < 1 or not isinstance(value, int):
                raise ValueError("C must be an integer greater or equal than 1")

    @staticmethod
    def __check_epsilon_param(epsilon: list[float | int]):
        """
        Check the validity of the epsilon parameter.

        Parameters:
            * epsilon: `list[float | int]`. The epsilon parameter to check.

        Raises:
            TypeError: If epsilon is not a list.
            ValueError: If epsilon is empty or if some value in epsilon is not between 0 and 1.
        """

        if not isinstance(epsilon, list):
            raise TypeError("epsilon must be a list")

        if len(epsilon) == 0:
            raise ValueError("epsilon cannot be empty")

        for value in epsilon:
            if value < 0 or value > 1:
                raise ValueError("epsilon must be between 0 and 1")

    def __check_validity_params(self, alpha: Union[int, float], sigma: Union[int, float], param_grid: dict):
        """
        Check the validity of the parameter grid.

        Parameters:
            * param_grid: `dict`. The parameter grid to check.

        Raises:
            TypeError: If the parameter grid is not a dictionary.
            ValueError: If some parameter is not valid or has an invalid value.
        """

        # Check if alpha and sigma are valid
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        if sigma < 1:
            raise ValueError("sigma must be greater or equal than 1")

        # Check if the parameter grid is a dictionary or if it is empty
        if not isinstance(param_grid, dict):
            raise TypeError("param_grid must be a dictionary")

        if not param_grid:
            raise ValueError("param_grid cannot be empty")

        # Check if the parameters are valid
        valid_params = ['m', 'R', 'C', 'G', 'epsilon']
        for key in param_grid.keys():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}. Valid parameters are: {valid_params}")

            if key == "m":
                self.__check_m_param(param_grid[key])

            if key == "R" or key == "G":
                self.__check_R_or_G_param(param_grid[key])

            if key == "C":
                self.__check_C_param(param_grid[key])

            if key == "epsilon":
                self.__check_epsilon_param(param_grid[key])

    @staticmethod
    def fit_single_instance(params):
        """
        Fit a single instance of the DRFL algorithm with a given set of parameters.

        Parameters:
            params (dict): A dictionary containing the parameters for a single DRFL instance.

        Returns:
            A dictionary containing the parameters used and the results of the DRFL fitting process.
        """

        m, R, C, G, epsilon, alpha, sigma, time_series, target_centroids = params
        drfl = DRFL(m=m, R=R, C=C, G=G, epsilon=epsilon)
        drfl.fit(time_series)
        estimated_distance = drfl.estimate_distance(target_centroids, alpha=alpha, sigma=sigma)
        return {"m": m, "R": R, "C": C, "G": G, "epsilon": epsilon, "estimated_distance": estimated_distance}

    def fit(self, time_series: pd.Series, target_centroids: list[list]):
        """
        Fit the DRFL algorithm to the time series data using multiple sets of parameters in parallel.

        Parameters:
            time_series (pd.Series): The time series data to analyze.
            target_centroids (list[list]): List of target centroids to compare with the discovered routines.
        """

        # set the fitted parameter to true
        self.__fitted = True

        # Prepare the list with all combinations of parameters to fit the DRFL instances
        all_params = list(product(
            [self.__param_grid.get('m', [3])],
            self.__param_grid.get('R', [2]),
            self.__param_grid.get('C', [3]),
            self.__param_grid.get('G', [4]),
            self.__param_grid.get('epsilon', [1]),
            [self.__alpha],
            [self.__sigma],
            [time_series],
            [target_centroids]
        ))

        # Use ProcessPoolExecutor to fit DRFL instances in parallel
        with ProcessPoolExecutor(max_workers=self.__n_jobs) as executor:
            results = list(executor.map(self.fit_single_instance, all_params))

        self.__results = results

    def cv_results(self) -> pd.DataFrame:
        """
        Return the cross-validation results after fitting the DRFL instances.

        Returns:
            A pandas DataFrame containing the results of the parallel search sorted by distance.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Examples:
            >>> from DRFL import ParallelSearchDRFL
            >>> import pandas as pd

            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> param_grid = {
            ...     "m": [3],
            ...     "R": [1, 2, 3, 4, 5, 6],
            ...     "C": [1, 2, 3, 4, 5, 6],
            ...     "G": [1, 2, 3, 4, 5, 6],
            ...     "epsilon": [0.5, 1],
            ... }
            >>> search = ParallelSearchDRFL(n_jobs=2, alpha=0.5, sigma=3, param_grid=param_grid)
            >>> search.fit(time_series)
            >>> print(search.cv_results())
            ... m R  C  G  epsilon  estimated_distance
            ... 3 1  3  4   1.0     0.0
            ... 3 1  2  5   1.0     0.0
            ... 3 2  1  5   1.0     0.0
            ... 3 2  2  5   1.0     0.0
            ...
        """

        # Check if the model has been fitted
        if not self.__fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        # Sort the results by estimated distance
        results = pd.DataFrame(self.__results).sort_values(by="estimated_distance")
        return pd.DataFrame(results)

    def best_params(self) -> dict:
        """
        Return the best parameters found during the search.

        Returns:
            `dict`. A dictionary containing the best parameters found during the search.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Examples:
            >>> from DRFL import ParallelSearchDRFL
            >>> import pandas as pd
            >>> time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
            >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
            >>> param_grid = {
            ...     "m": [3],
            ...     "R": [1, 2, 3, 4, 5, 6],
            ...     "C": [1, 2, 3, 4, 5, 6],
            ...     "G": [1, 2, 3, 4, 5, 6],
            ...     "epsilon": [0.5, 1],
            ... }
            >>> search = ParallelSearchDRFL(n_jobs=2, alpha=0.5, sigma=3, param_grid=param_grid)
            >>> search.fit(time_series)
            >>> print(search.best_params())
            >>> {'m': 3, 'R': 1, 'C': 3, 'G': 4, 'epsilon': 1}
        """

        if not self.__fitted:
            raise RuntimeError("The model has not been fitted yet. Please call the fit method before using this method")

        results = self.cv_results()
        return results.iloc[0].to_dict()
