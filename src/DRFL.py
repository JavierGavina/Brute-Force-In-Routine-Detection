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
import datetime

import numpy as np
import pandas as pd
from structures import Subsequence, Sequence, Cluster, Routines

from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


class DRFL:
    """
    Discovering Routines of Fixed Length.

    This class allows to discover routines of fixed length in a time series. The algorithm is based on the paper "An incremental algorithm for discovering routine behaviors from smart meter data" by Jin Wang, Rachel Cardell-Oliver and Wei Liu.

    The algorithm is based on the following steps:

            * Extract subsequences of fixed length from the time series.
            * Group the subsequences into clusters based on their magnitude and maximum absolute distance.
            * Filter the clusters based on their frequency.
            * Test and handle overlapping clusters.

    The algorithm is implemented in the class DRFL, which has the following methods and parameters:

    Parameters:
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
                  title_fontsize: `Optional[int]`. Size of the title plot.
                  ticks_fontsize: `Optional[int]`. Size of the ticks.
                  labels_fontsize: `Optional[int]`. Size of the labels.
                  figsize: `Optional[tuple[int, int]]`. Size of the figure.
                  xlim: `Optional[tuple[int, int]]`. Limit of the x axis with starting points.
                  save_dir: `Optional[str]`. Directory to save the plot.

    Examples:

        >>> from DRFL import DRFL
        >>> import pandas as pd

        >>> time_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
        >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=0.5)
        >>> drfl.fit(time_series)
        >>> drfl.show_results()
        >>> drfl.plot_results()
        ```
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
        self.time_series: pd.Series = None

    @staticmethod
    def __check_type_time_series(time_series: pd.Series) -> None:
        """
        Check the type of the time series.

        :param time_series: pd.Series. Temporal data
        :raises TypeError: If the time series is not a pandas Series.
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

    @staticmethod
    def __IsMatch(S1: 'Subsequence', S2: Union[np.ndarray, Subsequence], R: int | float) -> bool:
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
            >>> drfl.__obtain_keep_indices(0.5)
            [0, 1]

            >>> S1 = Subsequence(instance=np.array([1, 2, 3]), date=datetime.date(2024, 1, 1), starting_point=0)
            >>> S2 = Subsequence(instance=np.array([2, 3, 4]), date=datetime.date(2024, 1, 2), starting_point=1)
            >>> cluster1 = Cluster(centroid=S1, instances=Sequence(subsequence=S1))
            >>> cluster2 = Cluster(centroid=S2, instances=Sequence(subsequence=S2))
            >>> drfl = DRFL(m=3, R=2, C=3, G=4, epsilon=1.0)
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

    def show_results(self) -> None:
        """
        Displays the discovered routines after fitting the model to the time series data.

        This method prints out detailed information about each discovered routine, including the centroid of each cluster, the subsequence instances forming the routine, and the dates/times these routines occur.

        Note:
            This method should be called after the `fit` method to ensure that routines have been discovered and are ready to be displayed.
        """

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
        return self.__routines

    def plot_results(self, title_fontsize: Optional[int] = None,
                     xticks_fontsize: Optional[int] = None, yticks_fontsize: Optional[int] = None,
                     labels_fontsize: Optional[int] = None, figsize: Optional[tuple[int, int]] = (30, 10),
                     linewidth_bars: int = 1.5, xlim: Optional[tuple[int, int]] = None,
                     save_dir: Optional[str] = None) -> None:

        """
        This method uses matplotlib to plot the results of the algorithm. The plot shows the time series data with vertical dashed lines indicating the start of each discovered routine. The color of each routine is determined by the order in which they were discovered, and a legend is displayed to identify each routine.

        Parameters:
            * title_fontsize: `Optional[int]`. Size of the title plot.
            * xticks_fontsize: `Optional[int]`. Size of the xticks.
            * yticks_fontsize: `Optional[int]`. Size of the yticks.
            * labels_fontsize: `Optional[int]`. Size of the labels.
            * figsize: `Optional[tuple[int, int]]`. Size of the figure.
            * linewidth_bars: `Optional[int]`. Width of the bars in the plot.
            * xlim: `Optional[tuple[int, int]]`. Limit of the x axis with starting points.
            * save_dir: `Optional[str]`. Directory to save the plot.

        Notes:
           This method has to be executed after the fit method to ensure that routines have been discovered and are ready to be displayed.
        """

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
        xlim = xlim or (0, len(ts)-1)

        # Get the starting points of each routine
        start_points = [cluster.get_starting_points() for cluster in self.__routines]

        # For each routine, create a subplot and plot the routine
        for row, routine in enumerate(start_points):
            plt.subplot(N_rows, 1, row + 1)

            # Initialize the color of each data point in the time series as gray
            colors = ["gray"] * len(ts)

            # Set the title and x-label of the subplot
            plt.title(f'Routine {row + 1}', fontsize=title_fontsize or 20)
            plt.xlabel("Starting Points", fontsize=labels_fontsize or 20)

            # For each starting point in the routine, plot a vertical line and change the color of the data points in the routine
            for sp in routine:
                if xlim[0] <= sp <= xlim[1]:
                    plt.axvline(x=sp, color=base_colors[row], linestyle="--")
                    for j in range(self.m):
                        if sp + j <= xlim[1]:
                            plt.text(sp + j - 0.05, self.time_series[sp + j] - 0.8, f"{ts[sp + j]}", fontsize=20,
                                     backgroundcolor="white", color=base_colors[row])
                            colors[sp + j] = base_colors[row]

            # Plot the time series data as a bar plot
            plt.bar(x=np.arange(0, len(ts)), height=ts, color=colors, edgecolor="black", linewidth=linewidth_bars)

            # Set the ticks on the x-axis
            plt.xticks(ticks=np.arange(xlim[0], xlim[1]+1), labels=np.arange(xlim[0], xlim[1]+1),
                       fontsize=xticks_fontsize or 20)
            plt.yticks(fontsize=yticks_fontsize or 20)

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
