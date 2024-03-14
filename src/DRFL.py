import datetime

import numpy as np
import pandas as pd
from structures import Subsequence, Sequence, Cluster, Routines

from typing import Union
import matplotlib.pyplot as plt
import random


class DRFL:
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
    def __check_type_time_series(time_series: pd.Series):
        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")

        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise TypeError("time_series index must be a pandas DatetimeIndex")

    @staticmethod
    def __minimum_distance_index(distances: Union[np.ndarray, list]) -> int:
        return np.argmin(distances)

    def __extract_subsequence(self, time_series: pd.Series, t: int):
        """
        :param time_series: Temporal data
        :param t: Temporary Instance
        """
        window = time_series[t:t + self.m]
        subsequence = Subsequence(instance=window.values,
                                  date=time_series.index[t],
                                  starting_point=t)
        self.sequence.add_sequence(subsequence)

    def __IsMatch(self, S1: 'Subsequence', S2: Union[Subsequence | np.ndarray]) -> bool:
        if not isinstance(S1, Subsequence):
            raise TypeError("S1 must be instance of Subsequence")

        if isinstance(S2, Subsequence) or isinstance(S2, np.ndarray):
            return S1.Distance(S2) <= self.R

        raise TypeError("S2 must be instance of Subsequence or np.ndarray")

    def __NotTrivialMatch(self, subsequence: Subsequence, cluster: Cluster, start: int):
        if not isinstance(subsequence, Subsequence) or not isinstance(cluster, Cluster):
            raise TypeError("subsequence and cluster must be instances of Subsequence and Cluster respectively")

        if not self.__IsMatch(S1=subsequence, S2=cluster.centroid):
            return False

        for end in cluster.get_starting_points():
            for t in reversed(range(start + 1, end)):
                if self.__IsMatch(S1=subsequence, S2=self.sequence.get_by_starting_point(t)):
                    return False

        return True

    def __SubGroup(self):
        routines = Routines(Cluster(centroid=self.sequence[0].get_instance(),
                                    instances=Sequence(subsequence=self.sequence[0])))

        for i in range(1, len(self.sequence)):
            if self.sequence[i].Magnitude() > self.G:
                distances = [self.sequence[i].Distance(routines[j].centroid) for j in range(len(routines))]
                j_hat = self.__minimum_distance_index(distances)
                if self.__NotTrivialMatch(subsequence=self.sequence[i], cluster=routines[j_hat], start=i):
                    # Append new Sequence on the instances of Bm_j
                    routines[j_hat].add_instance(self.sequence[i])

                    # Update center of the cluster
                    routines[j_hat].update_centroid()

                else:
                    # create a new cluster//routine
                    new_cluster = Cluster(centroid=self.sequence[i].get_instance(),
                                          instances=Sequence(subsequence=self.sequence[i]))

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

    def __OLTest(self, cluster1: Cluster, cluster2: Cluster, epsilon: float):
        """
        Overlap testing for two lists of subsequences.
        :param cluster1: The first cluster of subsequences.
        :param cluster2: The second cluster of subsequences.
        :param epsilon: The overlap parameter.
        :return: keep tag for both clusters indicating which to keep.
        """
        N = 0  # Number of overlaps
        for S_i in cluster1.get_sequences():
            auxIndex_i = None
            for idx, sequence in enumerate(cluster1.get_sequences()):
                if np.array_equal(sequence.get_instance(), S_i.get_instance()):
                    auxIndex_i = cluster1.get_starting_points()[idx]
                    break
            S_i_info = Subsequence(instance=S_i.get_instance(), date=datetime.datetime.now(), starting_point=auxIndex_i)
            for S_j in cluster2.get_sequences():
                auxIndex_j = None
                for idx, sequence in enumerate(cluster2.get_sequences()):
                    if np.array_equal(sequence, S_j):
                        auxIndex_j = cluster2.get_starting_points()[idx]
                        break
                S_j_info = Subsequence(instance=S_j.get_instance(), date=datetime.datetime.now(),
                                       starting_point=auxIndex_j)
                if self.__IsOverlap(S_i_info, S_j_info):
                    N += 1
                    break  # Only need to find one overlap per S_i

        min_len = min(len(cluster1), len(cluster2))
        if N > epsilon * min_len:
            mag_cluster1 = sum([seq.Magnitude() for seq in cluster1.get_sequences()])
            mag_cluster2 = sum([seq.Magnitude() for seq in cluster2.get_sequences()])
            if len(cluster1) > len(cluster2) or (len(cluster1) == len(cluster2) and mag_cluster1 > mag_cluster2):
                return True, False
            else:
                return False, True

        return True, True  # If overlap is not significant, keep both

    def fit(self, time_series):
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

    def show_results(self):
        print("Routines detected: ", len(self.routines))
        print("_" * 50)
        for i, b in enumerate(self.routines):
            print(f"Centroid {i + 1}: {b.centroid}")
            print(f"Routine {i + 1}: {b.get_sequences().get_subsequences()}")
            print(f"Date {i + 1}: {b.get_dates()}")
            print("\n", "-" * 50, "\n")

    def get_results(self):
        return self.routines

    def plot_results(self, title: str = None, xlabel: str = None,
                     ylabel: str = None, figsize: tuple[int, int] = None,
                     xlim: tuple[datetime.date, datetime.date] = None,
                     ylim: tuple[int | float, int | float] = None,
                     xticklabels_rotation: int = None, yticklabels_rotation: int = None):

        # Plot the results
        colors = ["lightblue", "lightgreen", "lightyellow", "lightblack", "lightorange", "lightpurple", "lightpink",
                  "red", "blue", "green", "yellow", "black", "orange", "purple", "pink", "brown", "grey",
                  "lightbrown", "lightgrey", "lightred", "lightblue", "lightgreen", "lightyellow", "lightblack",
                  "lightorange", "lightpurple", "lightpink",
                  "red", "blue", "green", "yellow", "black", "orange", "purple", "pink", "brown", "grey"]

        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure(figsize=(20, 10))

        # Create a color list for the bars
        bar_colors = ['gray'] * len(self.time_series)

        for idx, cluster in enumerate(self.routines):
            for i in cluster.get_dates():
                # Find the indices of the bars between i and i+7
                bar_indices = (self.time_series.index >= i) & (self.time_series.index < i + pd.Timedelta(days=self.m))
                # Color the bars between i and i+7 with the corresponding color
                for j, is_colored in enumerate(bar_indices):
                    if is_colored:
                        bar_colors[j] = colors[idx]

        # Create a bar plot with the specified colors
        plt.bar(self.time_series.index, self.time_series.values, color=bar_colors)

        # Set the title, xlabel, and ylabel
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        # Set the x and y limits
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # Set the x and y ticks
        if xticklabels_rotation is not None:
            plt.xticks(rotation=xticklabels_rotation)
        if yticklabels_rotation is not None:
            plt.yticks(rotation=yticklabels_rotation)

        plt.show()

# if __name__ == "__main__":
#     # PARAMS
#     target_routine_1 = [20, 50, 70, 30]
#     target_routine_2 = [30, 30, 60, 30]
#     noise_threshold_minutes = 4
#     T_max = 21
#     idx_routine1 = [0, 1, 3, 5, 6, 8, 10, 11, 16, 17, 20]
#
#
#     def randomized_routine(routine, noise_threshold):
#         return [random.randint(x - noise_threshold, x + noise_threshold) for x in routine]
#
#
#     def get_colors(idx_routine):
#         colores = []
#         for x in range(T_max):
#             if x in idx_routine:
#                 for y in range(len(target_routine_1)):
#                     colores.append("red")
#             else:
#                 for y in range(len(target_routine_1)):
#                     colores.append("blue")
#         return colores
#
#
#     time_series = []
#     for x in range(T_max):
#         if x in idx_routine1:
#             random_1 = randomized_routine(target_routine_1, noise_threshold_minutes)
#             for y in random_1:
#                 time_series.append(y)
#         else:
#             random_2 = randomized_routine(target_routine_2, noise_threshold_minutes)
#             for y in random_2:
#                 time_series.append(y)
#
#     time_series = np.array(time_series)
#
#     # Plotting the bar chart with vertical lines every 4 bars
#     plt.figure(figsize=(T_max, 2))
#     bars = plt.bar(x=[x for x in range(len(time_series))], height=time_series, color=get_colors(idx_routine1))
#
#     # Draw a vertical line every four bars
#     for i in range(0, len(time_series), len(target_routine_1)):
#         plt.axvline(x=i - 0.5, color='grey', linestyle='--')
#     plt.xticks(ticks=[x for x in range(len(time_series))],
#                labels=pd.date_range(start="2024-01-01", periods=len(time_series)),
#                rotation=90)
#     plt.show()
#
#     time_series = pd.DataFrame(
#         {
#             "Date": pd.date_range(start="2024-01-01", periods=len(time_series)),
#             "Time-Series": time_series
#         }
#     ).set_index("Date")["Time-Series"]
#
#     routine_detector = DRFL(m=4, R=8, C=5, G=50, epsilon=0.5)
#     routine_detector.fit(time_series)
#     routine_detector.show_results()
