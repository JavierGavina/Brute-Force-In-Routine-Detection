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
    """

    def __init__(self, m: int, R: float | int, C: int, G: float | int, epsilon: float):
        """
        :param m: Length of each secuence
        :param R: Distance threshold
        :param C: Frequency threshold
        :param G: Magnitude threshold
        :param epsilon: Overlap Parameter
        """

        self.m: int = m
        self.R: int | float = R
        self.C: int = C
        self.G: int | float = G
        self.epsilon: float = epsilon
        self.routines: Routines = Routines()
        self.sequence: Sequence = Sequence()
        self.time_series: pd.Series

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

    @staticmethod
    def __IsMatch(S1: 'Subsequence', S2: np.ndarray | Subsequence, R: int | float) -> bool:
        """
        Check if two subsequences match by checking if the distance between them is lower than the threshold distance parameter R.

        :param S1: Subsequence. The first subsequence.
        :param S2: np.array or Subsequence. The second subsequence.
        :param R: int or float. The threshold distance parameter.
        :return: bool. True if the distance between the subsequences is lower than the threshold distance parameter R, False otherwise.
        :raises TypeError: If S1 is not an instance of Subsequence or S2 is not an instance of Subsequence or np.ndarray.
        """
        if not isinstance(S1, Subsequence):
            raise TypeError("S1 must be instance of Subsequence")

        if isinstance(S2, Subsequence) or isinstance(S2, np.ndarray):
            return S1.Distance(S2) <= R

        raise TypeError("S2 must be instance of Subsequence or np.ndarray")

    def __NotTrivialMatch(self, subsequence: Subsequence, cluster: Cluster, start: int, R: int | float) -> bool:
        """
        Check if a subsequence is not a trivial match with any of the instances of the cluster.

        :param subsequence: Subsequence. The subsequence to check.
        :param cluster: Cluster. The cluster to check.
        :param start: int. Starting point of the subsequence.
        :param R: int or float. The threshold distance parameter.
        :return: bool. True if the subsequence is not a trivial match with any of the instances of the cluster, False otherwise.
        :raises TypeError: If subsequence is not an instance of Subsequence or cluster is not an instance of Cluster.
        """
        if not isinstance(subsequence, Subsequence) or not isinstance(cluster, Cluster):
            raise TypeError("subsequence and cluster must be instances of Subsequence and Cluster respectively")

        if not self.__IsMatch(S1=subsequence, S2=cluster.centroid, R=R):
            return False

        for end in cluster.get_starting_points():
            for t in reversed(range(start + 1, end)):
                if self.__IsMatch(S1=subsequence, S2=self.sequence.get_by_starting_point(t), R=R):
                    return False

        return True

    # def __SubGroup(self, R: float | int, C: int, G: float | int) -> Routines:
    #     """
    #     Group the subsequences into clusters based on their magnitude and maximum absolute distance.
    #     The steps that follow this algorithm are:
    #         * Create a new cluster with the first subsequence.
    #         * For each subsequence, check if it is not a trivial match with any of the instances within the cluster.
    #         * If it is not a trivial match, append new Sequence on the instances of the cluster.
    #         * If it is a trivial match, create a new cluster.
    #         * Filter the clusters by frequency.
    #
    #     :param R: float or int. Distance threshold.
    #     :param C: int. Frequency threshold.
    #     :param G: float or int. Magnitude threshold.
    #     :return: Routines. The clusters of subsequences.
    #     """
    #
    #     routines = Routines(Cluster(centroid=self.sequence[0].get_instance(),
    #                                 instances=Sequence(subsequence=self.sequence[0])))
    #
    #     for i in range(1, len(self.sequence)):
    #
    #         # Check if the magnitude of the subsequence is greater than the threshold magnitude parameter G
    #         if self.sequence[i].Magnitude() > G:
    #
    #             # Estimate all the distances between the subsequence and all the centroids of the clusters
    #             distances = [self.sequence[i].Distance(routines[j].centroid) for j in range(len(routines))]
    #
    #             # Get the index of the minimum distance to the centroid
    #             j_hat = self.__minimum_distance_index(distances)
    #
    #             # Check if the subsequence is not a trivial match with any of the instances within the cluster
    #             # if self.__NotTrivialMatch(subsequence=self.sequence[i], cluster=routines[j_hat], start=i, R=R):
    #             if self.__IsMatch(S1=self.sequence[i], S2=routines[j_hat].centroid, R=R):
    #                 # Append new Sequence on the instances of Bm_j
    #                 routines[j_hat].add_instance(self.sequence[i])
    #
    #                 # Update center of the cluster
    #                 routines[j_hat].update_centroid()
    #             else:
    #
    #                 # create a new cluster//routine
    #                 new_cluster = Cluster(centroid=self.sequence[i].get_instance(),
    #                                       instances=Sequence(subsequence=self.sequence[i]))
    #
    #                 # Add the new cluster to the list of clusters // new routine
    #                 routines.add_routine(new_cluster)
    #
    #     # Filter by frequency
    #     to_drop = [k for k in range(len(routines)) if len(routines[k]) < C]
    #     filtered_routines = routines.drop_indexes(to_drop)
    #
    #     return filtered_routines

    def __SubGroup(self, R: float | int, C: int, G: float | int) -> Routines:
        """
        Group the subsequences into clusters based on their magnitude and maximum absolute distance.
        The steps that follow this algorithm are:
            * Create a new cluster with the first subsequence.
            * For each subsequence, check if it is not a trivial match with any of the instances within the cluster.
            * If it is not a trivial match, append new Sequence on the instances of the cluster.
            * If it is a trivial match, create a new cluster.
            * Filter the clusters by frequency.

        :param R: float or int. Distance threshold.
        :param C: int. Frequency threshold.
        :param G: float or int. Magnitude threshold.
        :return: Routines. The clusters of subsequences.
        """

        routines = Routines()

        for i in range(len(self.sequence)):
            # Check if the magnitude of the subsequence is greater than the threshold magnitude parameter G
            if self.sequence[i].Magnitude() >= G:
                if routines.is_empty():
                    routines.add_routine(Cluster(centroid=self.sequence[i].get_instance(),
                                                 instances=Sequence(subsequence=self.sequence[i])))
                    continue

                # Estimate all the distances between the subsequence and all the centroids of the clusters
                distances = [self.sequence[i].Distance(routines[j].centroid) for j in range(len(routines))]

                # Get the index of the minimum distance to the centroid
                j_hat = self.__minimum_distance_index(distances)

                # Check if the subsequence is not a trivial match with any of the instances within the cluster
                # if self.__NotTrivialMatch(subsequence=self.sequence[i], cluster=routines[j_hat], start=i, R=R):
                if self.__IsMatch(S1=self.sequence[i], S2=routines[j_hat].centroid, R=R):
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
        to_drop = [k for k in range(len(routines)) if len(routines[k]) < C]
        filtered_routines = routines.drop_indexes(to_drop)

        return filtered_routines

    @staticmethod
    def __IsOverlap(S_i: Subsequence, S_j: Subsequence):
        """
        Check if two subsequences overlap.

        Parameters:
            * S_i: `Subsequence`. The first subsequence with starting point i.
            * S_j: `Subsequence`. The second subsequence with starting point j.

        Returns:
             `True` if they overlap, `False` otherwise.
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
        """

        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be between 0 and 1")

        # Prepare to test and handle overlapping clusters
        keep_indices = set(range(len(self.routines)))  # Initially, assume all clusters are to be kept

        for i in range(len(self.routines) - 1):
            for j in range(i + 1, len(self.routines)):
                if i in keep_indices and j in keep_indices:  # Process only if both clusters are still marked to keep
                    keep_i, keep_j = self.__OLTest(self.routines[i], self.routines[j], epsilon)

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
        """

        self.__check_type_time_series(time_series)
        self.time_series = time_series
        for i in range(len(self.time_series) - self.m + 1):
            self.__extract_subsequence(self.time_series, i)

        # Group the subsequences into clusters based on their magnitude and
        # maximum absolute distance and filter the clusters based on their frequency
        self.routines = self.__SubGroup(R=self.R, C=self.C, G=self.G)

        # Obtain the indices of the clusters to keep based on the overlap test
        keep_indices = self.__obtain_keep_indices(self.epsilon)

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
        return self.routines

    def plot_results(self, title_fontsize: Optional[int] = None, ticks_fontsize: Optional[int] = None,
                     labels_fontsize: Optional[int] = None, figsize: Optional[tuple[int, int]] = (30, 10),
                     xlim: Optional[tuple[int, int]] = None, save_dir: Optional[str] = None) -> None:

        """
        This method uses matplotlib to plot the results of the algorithm. The plot shows the time series data with vertical dashed lines indicating the start of each discovered routine. The color of each routine is determined by the order in which they were discovered, and a legend is displayed to identify each routine.

        Parameters:
            * title_fontsize: `Optional[int]`. Size of the title plot.
            * ticks_fontsize: `Optional[int]`. Size of the ticks.
            * labels_fontsize: `Optional[int]`. Size of the labels.
            * figsize: `Optional[tuple[int, int]]`. Size of the figure.
            * xlim: `Optional[tuple[int, int]]`. Limit of the x axis with starting points.
            * save_dir: `Optional[str]`. Directory to save the plot.

        Notes:
           This method has to be executed after the fit method to ensure that routines have been discovered and are ready to be displayed.
        """

        # Generate a color map for the routines
        base_colors = cm.rainbow(np.linspace(0, 1, len(self.routines)))

        # Convert the time series data to a numpy array for easier manipulation
        ts = np.array(self.time_series)

        # Create a new figure with the specified size
        plt.figure(figsize=figsize)

        # Get the number of routines and the maximum value in the time series
        N_rows = len(self.routines)
        maximum = max(ts)

        # Get the starting points of each routine
        start_points = [cluster.get_starting_points() for cluster in self.routines]

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
                plt.axvline(x=sp, color=base_colors[row], linestyle="--")
                for j in range(self.m):
                    plt.text(sp + j - 0.05, self.time_series[sp + j] - 0.8, f"{ts[sp + j]}", fontsize=20,
                             backgroundcolor="white", color=base_colors[row])
                    colors[sp + j] = base_colors[row]

            # Plot the time series data as a bar plot
            plt.bar(x=np.arange(0, len(ts)), height=ts, color=colors)

            # Plot a horizontal line at the magnitude threshold
            plt.axhline(y=self.G, color="red", linestyle="--")

            # Set the limits of the x-axis and y-axis
            plt.xlim(xlim or (-1, len(ts)))
            plt.ylim(0, maximum + 1)

            # Set the ticks on the x-axis
            plt.xticks(ticks=np.arange(0, len(ts)), labels=np.arange(0, len(ts)), fontsize=ticks_fontsize or 20)

            # Adjust the layout of the plot
            plt.tight_layout()

        # If a directory is provided, save the figure to the directory
        if save_dir:
            plt.savefig(save_dir)

        # Display the plot
        plt.show()


if __name__ == "__main__":
    def plot_groundtruth(time_series, start_points, m, G=4, save_dir=None):
        base_colors = cm.rainbow(np.linspace(0, 1, len(start_points)))
        ts = np.array(time_series)
        plt.figure(figsize=(30, 10))
        N_rows = len(start_points)
        maximum = max(time_series)
        for row, routine in enumerate(start_points):
            all_colors = ["gray"] * len(time_series)
            plt.subplot(N_rows, 1, row + 1)
            plt.title(f"Target Routine {row + 1}", fontsize=20)
            for start_point in routine:
                plt.axvline(x=start_point, color=base_colors[row], linestyle='--')
                for j in range(m):
                    plt.text(start_point + j - 0.05, time_series[start_point + j] - 0.8,
                             f"{time_series[start_point + j]}",
                             fontsize=20, color=base_colors[row], backgroundcolor="white")
                    all_colors[start_point + j] = base_colors[row]
            plt.bar(x=[x for x in range(len(ts))], height=ts, color=all_colors)
            plt.axhline(y=G, color='red', linestyle=':')
            plt.xticks(ticks=[x for x in range(len(ts))], labels=[x for x in range(len(ts))], fontsize=20)
            plt.ylim((0, maximum + 1))
            plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir)

        plt.show()


    target_start_points = [[0, 6, 11], [1, 7, 12], [2, 8, 13]]
    time_series = np.array([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    time_series = pd.DataFrame(
        {
            "Date": pd.date_range(start="2024-01-01", periods=len(time_series)),
            "Time-Series": time_series
        }
    ).set_index("Date")["Time-Series"]

    # ----------------GROUNDTRUTH (APPROXIMATION)-------------------
    plot_groundtruth(time_series, target_start_points, 3, 4)

    # ----------------WITH OVERLAPPING EPSILON=1--------------------
    routines = DRFL(m=3, G=4, R=2, C=3, epsilon=1)
    routines.fit(time_series)
    routines.show_results()
    routines.plot_results()

    # ----------------WITHOUT OVERLAPPING EPSILON=0.5----------------
    routines = DRFL(m=3, G=4, R=2, C=3, epsilon=0.5)
    routines.fit(time_series)
    routines.show_results()

    routines.plot_results()
