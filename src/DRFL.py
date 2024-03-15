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
            - title: Title of the plot.
            - title_fontsize: Font size of the title.
            - xlabel: Label of the x axis.
            - ylabel: Label of the y axis.
            - ticks_fontsize: Font size of the ticks.
            - labels_fontsize: Font size of the labels.
            - figsize: Size of the figure.
            - xlim: Limit of the x axis.
            - ylim: Limit of the y axis.
            - xticklabels_rotation: Rotation of the x axis labels.
            - yticklabels_rotation: Rotation of the y axis labels.
            - show_legend: Show the legend.
            - legend_title: Title of the legend.
            - legend_title_fontsize: Font size of the legend title.
            - legend_labels_fontsize: Font size of the legend labels.

"""
import datetime

import numpy as np
import pandas as pd
from structures import Subsequence, Sequence, Cluster, Routines

from typing import Union
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
    """

    def __init__(self, m: int, R: float | int, C: int, G: float | int, epsilon: float):
        """
        :param m: Length of each secuence
        :param R: Distance threshold
        :param C: Frequency threshold
        :param G: Magnitude threshold
        :param epsilon: Overlap Parameter
        """

        self.m = m
        self.R = R
        self.C = C
        self.G = G
        self.epsilon = epsilon
        self.routines = Routines()
        self.sequence = Sequence()
        self.time_series = None

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

        :param distances: np.array or list. List of distances.
        :return: int. Index of the minimum distance.
        """

        return np.argmin(distances)

    def __extract_subsequence(self, time_series: pd.Series, t: int) -> None:
        """
        Extract a subsequence from the time series and adds the subsequence to Sequence object.

        :param time_series: pd.Series. Temporal data.
        :param t: int. Starting point of the subsequence.
        """
        window = time_series[t:t + self.m]
        subsequence = Subsequence(instance=window.values,
                                  date=time_series.index[t],
                                  starting_point=t)
        self.sequence.add_sequence(subsequence)

    def __IsMatch(self, S1: 'Subsequence', S2: np.ndarray | Subsequence) -> bool:
        """
        Check if two subsequences match by checking if the distance between them is lower than the threshold distance parameter R.

        :param S1: Subsequence. The first subsequence.
        :param S2: np.array or Subsequence. The second subsequence.
        :return: bool. True if the distance between the subsequences is lower than the threshold distance parameter R, False otherwise.
        :raises TypeError: If S1 is not an instance of Subsequence or S2 is not an instance of Subsequence or np.ndarray.
        """
        if not isinstance(S1, Subsequence):
            raise TypeError("S1 must be instance of Subsequence")

        if isinstance(S2, Subsequence) or isinstance(S2, np.ndarray):
            return S1.Distance(S2) <= self.R

        raise TypeError("S2 must be instance of Subsequence or np.ndarray")

    def __NotTrivialMatch(self, subsequence: Subsequence, cluster: Cluster, start: int) -> bool:
        """
        Check if a subsequence is not a trivial match with any of the instances of the cluster.

        :param subsequence: Subsequence. The subsequence to check.
        :param cluster: Cluster. The cluster to check.
        :param start: int. Starting point of the subsequence.
        :return: bool. True if the subsequence is not a trivial match with any of the instances of the cluster, False otherwise.
        :raises TypeError: If subsequence is not an instance of Subsequence or cluster is not an instance of Cluster.
        """
        if not isinstance(subsequence, Subsequence) or not isinstance(cluster, Cluster):
            raise TypeError("subsequence and cluster must be instances of Subsequence and Cluster respectively")

        if not self.__IsMatch(S1=subsequence, S2=cluster.centroid):
            return False

        for end in cluster.get_starting_points():
            for t in reversed(range(start + 1, end)):
                if self.__IsMatch(S1=subsequence, S2=self.sequence.get_by_starting_point(t)):
                    return False

        return True

    def __SubGroup(self) -> Routines:
        """
        Group the subsequences into clusters based on their magnitude and maximum absolute distance.
        The steps that follow this algorithm are:
            * Create a new cluster with the first subsequence.
            * For each subsequence, check if it is not a trivial match with any of the instances of the cluster.
            * If it is not a trivial match, append new Sequence on the instances of the cluster.
            * If it is a trivial match, create a new cluster.
            * Filter the clusters by frequency.

        :return: Routines. The clusters of subsequences.
        """

        routines = Routines(Cluster(centroid=self.sequence[0].get_instance(),
                                    instances=Sequence(subsequence=self.sequence[0])))

        for i in range(1, len(self.sequence)):

            # Check if the magnitude of the subsequence is greater than the threshold magnitude parameter G
            if self.sequence[i].Magnitude() > self.G:

                # Estimate all the distances between the subsequence and all the centroids of the clusters
                distances = [self.sequence[i].Distance(routines[j].centroid) for j in range(len(routines))]

                # Get the index of the minimum distance to the centroid
                j_hat = self.__minimum_distance_index(distances)

                # Check if the subsequence is not a trivial match with any of the instances of the cluster
                if self.__NotTrivialMatch(subsequence=self.sequence[i], cluster=routines[j_hat], start=i):

                    # Append new Sequence on the instances of Bm_j
                    routines[j_hat].add_instance(self.sequence[i])

                    # Update center of the cluster
                    routines[j_hat].update_centroid()
                else:

                    # create a new cluster//routine
                    new_cluster = Cluster(centroid=self.sequence[i].get_instance(),
                                          instances=Sequence(subsequence=self.sequence[i]))

                    # Add the new cluster to the list of clusters // new routine
                    routines.add_routine(new_cluster)

        # Filter by frequency
        to_drop = [k for k in range(len(routines)) if len(routines[k]) < self.C]
        filtered_routines = routines.drop_indexes(to_drop)

        return filtered_routines

    @staticmethod
    def __IsOverlap(S_i: Subsequence, S_j: Subsequence):
        """
        Check if two subsequences overlap.
        :param S_i: The first subsequence with starting point i.
        :param S_j: The second subsequence with starting point j.
        :return: True if they overlap, False otherwise.
        """
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
            cluster1: `Cluster`. The first cluster involved in the overlap test.
            cluster2: `Cluster`. The second cluster involved in the overlap test.
            epsilon: `float`. A threshold parameter that defines the minimum percentage of overlap required for considering an overlap significant. Values range from 0 to 1, where a higher value means a stricter criterion for significance.

        Returns:
            tuple[bool, bool]: A tuple containing two boolean values. The first value indicates whether
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
            mag_cluster1 = sum([seq.Magnitude() for seq in cluster1.get_sequences()])
            mag_cluster2 = sum([seq.Magnitude() for seq in cluster2.get_sequences()])

            # Keep the cluster with either more instances or, in a tie, the greater magnitude
            if len(cluster1) > len(cluster2) or (len(cluster1) == len(cluster2) and mag_cluster1 > mag_cluster2):
                return True, False
            else:
                return False, True
        else:
            # If overlap is not significant, propose to keep both clusters
            return True, True

    def fit(self, time_series: pd.Series) -> None:
        """
        Fits the time series data to the `DRFL` algorithm to discover routines.

        This method preprocesses the time series data, extracts subsequences, groups them into clusters, and finally filters and handles overlapping clusters to discover and refine routines.
        :param: time_series: `pd.Series`. The time series data to analyze. It should be a `pandas Series` object with a `DatetimeIndex`.
        :raises: `TypeError`: If the input time series is not a `pandas Series` or if its index is not a `DatetimeIndex`.
        """

        self.__check_type_time_series(time_series)
        self.time_series = time_series
        for i in range(len(self.time_series) - self.m):
            self.__extract_subsequence(self.time_series, i)

        self.routines = self.__SubGroup()

        # Prepare to test and handle overlapping clusters
        keep_indices = set(range(len(self.routines)))  # Initially, assume all clusters are to be kept

        for i in range(len(self.routines) - 1):
            for j in range(i + 1, len(self.routines)):
                if i in keep_indices and j in keep_indices:  # Process only if both clusters are still marked to keep
                    keep_i, keep_j = self.__OLTest(self.routines[i], self.routines[j], self.epsilon)

                    # Update keep indices based on OLTest outcome
                    if not keep_i:
                        keep_indices.remove(i)
                    if not keep_j:
                        keep_indices.remove(j)

        # Filter self.routines to keep only those clusters marked for keeping
        if len(self.routines) > 0:
            to_drop = [k for k in range(len(self.routines)) if k not in keep_indices]
            self.routines = self.routines.drop_indexes(to_drop)

    def show_results(self) -> None:
        """
        Displays the discovered routines after fitting the model to the time series data.

        This method prints out detailed information about each discovered routine, including the centroid of each cluster, the subsequence instances forming the routine, and the dates/times these routines occur.

        Note:
            This method should be called after the `fit` method to ensure that routines have been discovered and are ready to be displayed.
        """

        print("Routines detected: ", len(self.routines))
        print("_" * 50)
        for i, b in enumerate(self.routines):
            print(f"Centroid {i + 1}: {b.centroid}")
            print(f"Routine {i + 1}: {b.get_sequences().get_subsequences()}")
            print(f"Date {i + 1}: {b.get_dates()}")
            print("\n", "-" * 50, "\n")

    def get_results(self) -> Routines:
        """
        Returns the discovered routines as a `Routines` object.

        After fitting the model to the time series data, this method can be used to retrieve the discovered routines, encapsulated within a `Routines` object, which contains all the clusters (each representing a routine) identified by the algorithm.

        :return: `Routines`. The discovered routines as a `Routines` object.

        Note:
            The `Routines` object provides methods and properties to further explore and manipulate the discovered routines.
        """
        return self.routines

    def plot_results(self, title: str | None = None, title_fontsize: int | None = None,
                     xlabel: str = None, ylabel: str = None, ticks_fontsize: int | None = None,
                     labels_fontsize: int | None = None, figsize: tuple[int, int] = (20, 10),
                     xlim: tuple[datetime.date, datetime.date] | None = None,
                     ylim: tuple[int | float, int | float] | None = None,
                     xticklabels_rotation: int | None = None, yticklabels_rotation: int | None = None,
                     show_legend: bool = True, legend_title: str = None, legend_title_fontsize: int | None = None,
                     legend_labels_fontsize: int | None = None):

        """
        This method uses matplotlib to plot the results of the algorithm. The plot shows the time series data with vertical dashed lines indicating the start of each discovered routine. The color of each routine is determined by the order in which they were discovered, and a legend is displayed to identify each routine.

        :param title: str. Title of the plot.
        :param title_fontsize: int. Font size of the title.
        :param xlabel: str. Label of the x axis.
        :param ylabel: str. Label of the y axis.
        :param ticks_fontsize: int. Font size of the ticks.
        :param labels_fontsize: int. Font size of the labels.
        :param figsize: tuple[int, int]. Size of the figure.
        :param xlim: tuple[datetime.date, datetime.date]. Limit of the x axis.
        :param ylim: tuple[float | int, float | int]. Limit of the y axis.
        :param xticklabels_rotation: int. Rotation of the x axis labels.
        :param yticklabels_rotation: int. Rotation of the y axis labels.
        :param show_legend: bool = True. Show the legend.
        :param legend_title: str. Title of the legend.
        :param legend_title_fontsize: int. Font size of the legend title.
        :param legend_labels_fontsize: int. Font size of the legend labels.

        Notes:
           This method so be executed after the fit method to ensure that routines have been discovered and are ready to be displayed.
        """

        colors = cm.rainbow(np.linspace(0, 1, len(self.routines)))
        plt.figure(figsize=figsize)

        bar_colors = ['gray'] * len(self.time_series)
        for idx, cluster in enumerate(self.routines):
            for i in cluster.get_dates():
                bar_indices = (self.time_series.index >= i) & (self.time_series.index < i + pd.Timedelta(days=self.m))
                for j, is_colored in enumerate(bar_indices):
                    if is_colored:
                        plt.axvline(x=i, color=colors[idx], linestyle='--')
                        bar_colors[j] = colors[idx]

        plt.bar(self.time_series.index, self.time_series.values, color=bar_colors)

        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(xlabel, fontsize=labels_fontsize)
        plt.ylabel(ylabel, fontsize=labels_fontsize)

        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)

        plt.xticks(rotation=xticklabels_rotation or 0, fontsize=ticks_fontsize)
        plt.yticks(rotation=yticklabels_rotation or 0, fontsize=ticks_fontsize)

        if show_legend:
            legend_labels = [f'Routine {i + 1}' for i in range(len(self.routines))]
            patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
            plt.legend(handles=patches, loc='upper right', title=legend_title or '',
                       fontsize=legend_labels_fontsize or labels_fontsize,
                       title_fontsize=legend_title_fontsize or title_fontsize)

        plt.show()
