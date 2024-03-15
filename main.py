import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.DRFL import DRFL

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", type=str, default="data/activities-simulation.csv", help="Path to the data file")
argparser.add_argument("--dictionary_dir", type=str, default="data/dictionary_rooms.json",
                       help="Path to the dictionary file")
argparser.add_argument("--param_m", type=int, default=4, help="length of the subsequences")
argparser.add_argument("--param_R", type=int, default=10, help="least maximum distance between subsequences")
argparser.add_argument("--param_C", type=int, default=4, help="minimum number of matches of a routine")
argparser.add_argument("--param_G", type=int, default=60, help="minimum magnitude of a subsequence")
argparser.add_argument("--epsilon", type=float, default=0.5, help="minimum overlap percentage")


def process_sequence(sequence: list):
    return np.array([np.nan if x == "" else int(x) for x in sequence])


# Load the data
def load_data(data_dir: str):
    data = pd.DataFrame(columns=["Year", "Month", "Day", "Sequence"])
    with open(data_dir, "r") as file:
        # skip the header
        file.readline()
        for idx, line in enumerate(file.readlines()):  # Read each line
            text = line.rstrip("\n").split(",")
            sequence_processed = process_sequence(text[3:])
            data.loc[idx] = [int(text[0]), int(text[1]), int(text[2]), sequence_processed]

    return data


def obtain_correspondencies(json_dictionary_file: str):
    with open(json_dictionary_file, "r") as file:
        correspondences = json.load(file)
        # invert correspondences
        correspondences = {v: k for k, v in correspondences.items()}
    return correspondences


def feature_extraction(data: pd.DataFrame, correspondences: dict) -> pd.DataFrame:
    feat_extraction = data.copy()
    # Create a new column with the room name
    rooms_keys = correspondences.keys()
    for key in rooms_keys:
        if correspondences[key] != "room" and correspondences[key] != "dining-room":
            feat_extraction[f"N_{key}"] = (feat_extraction["Sequence"]
                                           .apply(lambda x: sum(x == key)))
    feat_extraction = feat_extraction.drop(columns="Sequence")
    return feat_extraction


def plot_feat_extraction_days(feat_extract: pd.DataFrame):
    columnas = feat_extract.columns[3:].tolist()
    aux = feat_extract.copy()
    aux["date"] = pd.to_datetime(aux[["Year", "Month", "Day"]])
    for col in columnas:
        key = int(col.replace("N_", ""))
        plt.bar(aux["date"], feat_extract[col], label=correspondencies[key])
    plt.legend()
    plt.show()


def plot_gym_hours(feat_extract: pd.DataFrame):
    columnas = feat_extract.columns[3:].tolist()
    aux = feat_extract.copy()
    aux["date"] = pd.to_datetime(aux[["Year", "Month", "Day"]])
    for col in columnas:
        key = int(col.replace("N_", ""))
        if correspondencies[key] == "gym":
            plt.bar(aux["date"], feat_extract[col], label=correspondencies[key])

    plt.xlim((pd.to_datetime("2024-02-01"), pd.to_datetime("2024-10-31")))
    plt.legend()
    plt.xticks(rotation=30)
    plt.show()


def get_time_series(feat_extract: pd.DataFrame, room: str):
    columnas = feat_extract.columns[3:].tolist()
    aux = feat_extract.copy()
    aux["date"] = pd.to_datetime(aux[["Year", "Month", "Day"]])
    aux = aux.set_index("date")
    for col in columnas:
        key = int(col.replace("N_", ""))
        if correspondencies[key] == room:
            time_series = aux[col]
            time_series.name = room
            return time_series


# class DRFL:
#     def __init__(self, m: int, R: float | int, C: int, G: float | int, epsilon: float):
#         """
#         :param m: Length of each secuence
#         :param R: Distance threshold
#         :param C: Frequency threshold
#         :param G: Magnitude threshold
#         :param epsilon: Overlap Parameter
#         """
#         self.m = m
#         self.R = R
#         self.C = C
#         self.G = G
#         self.epsilon = epsilon
#         self.Bm = None
#         self.Sm = []
#
#     @staticmethod
#     def __Magnitude(sequence: np.array):
#
#         return np.max(sequence)
#
#     @staticmethod
#     def __Distance(S1: np.ndarray, S2: np.ndarray) -> int | float:
#         """
#
#         :param S1: Left sequence to obtain distance
#         :param S2: Right sequence to obtain distance
#         :return: distance
#         """
#         return np.max(np.abs(S1 - S2))
#
#     @staticmethod
#     def __get_j_hat(distances: np.ndarray):
#         return np.argmin(distances)
#
#     @staticmethod
#     def __GetUpdatedCentre(Bmj) -> np.ndarray:
#         return np.mean(Bmj, axis=0)
#
#     def __extract_subsequence(self, time_series: pd.Series, t: int):
#         """
#         :param time_series: Temporal data
#         :param t: Temporary Instance
#         """
#         sequence = time_series[t:t + self.m]
#         new_params = {"Sequence": time_series[t:t + self.m].values, "Date": time_series.index[t], "auxIndex": t}
#         self.Sm.append(new_params)
#
#     def __getSequenceFromAuxIndex(self, auxIndex: int):
#         for seq in self.Sm:
#             if seq["auxIndex"] == auxIndex:
#                 return seq["Sequence"]
#
#     def __IsMatch(self, S1, S2, R):
#         return self.__Distance(S1, S2) <= R
#
#     def __NotTrivialMatch(self, sequence, cluster, start, R):
#         if not self.__IsMatch(sequence, cluster["Cent"], R):
#             return False
#
#         for end in cluster["auxIndex"]:
#
#             for t in reversed(range(start + 1, end)):
#                 if self.__IsMatch(sequence, self.__getSequenceFromAuxIndex(t), R):
#                     return False
#
#         return True
#
#     def __SubGroup(self, R, C, G):
#         routines = [{"Cent": self.Sm[0]["Sequence"], "Inst": [self.Sm[0]["Sequence"]],
#                      "Date": [self.Sm[0]["Date"]], "auxIndex": [self.Sm[0]["auxIndex"]]}]
#         for i in range(1, len(self.Sm)):
#             if self.__Magnitude(self.Sm[i]["Sequence"]) > G:
#                 distances = [self.__Distance(self.Sm[i]["Sequence"], routines[j]["Cent"]) for j in range(len(routines))]
#                 j_hat = self.__get_j_hat(distances)
#                 if self.__NotTrivialMatch(self.Sm[i]["Sequence"], routines[j_hat], i, R):
#                     # Append new Sequence on the instances of Bm_j
#                     routines[j_hat]["Inst"].append(self.Sm[i]["Sequence"])
#                     routines[j_hat]["Date"].append(self.Sm[i]["Date"])
#                     routines[j_hat]["auxIndex"].append(self.Sm[i]["auxIndex"])
#
#                     # Update center of the cluster
#                     routines[j_hat]["Cent"] = self.__GetUpdatedCentre(routines[j_hat]["Inst"])
#
#                 else:
#                     # create a new center
#                     routines.append({"Cent": self.Sm[i]["Sequence"], "Inst": [self.Sm[i]["Sequence"]],
#                                      "Date": [self.Sm[i]["Date"]], "auxIndex": [self.Sm[i]["auxIndex"]]})
#
#         # Filter by frequency
#         to_drop = [k for k in range(len(routines)) if len(routines[k]["Inst"]) < C]
#         filtered_routines = [value for idx, value in enumerate(routines) if not (idx in to_drop)]
#
#         return filtered_routines
#
#     def __IsOverlap(self, S_i, S_j):
#         """
#         Check if two subsequences overlap.
#         :param S_i: The first subsequence with starting point i.
#         :param S_j: The second subsequence with starting point j.
#         :return: True if they overlap, False otherwise.
#         """
#         start_i, p = S_i['auxIndex'], len(S_i['Sequence'])
#         start_j, q = S_j['auxIndex'], len(S_j['Sequence'])
#         return not ((start_i + p <= start_j) or (start_j + q <= start_i))
#
#     def __OLTest(self, cluster1, cluster2, epsilon):
#         """
#         Overlap testing for two lists of subsequences.
#         :param cluster1: The first cluster of subsequences.
#         :param cluster2: The second cluster of subsequences.
#         :return: keep tag for both clusters indicating which to keep.
#         """
#         N = 0  # Number of overlaps
#         for S_i in cluster1['Inst']:
#             auxIndex_i = None
#             for idx, sequence in enumerate(cluster1['Inst']):
#                 if np.array_equal(sequence, S_i):
#                     auxIndex_i = cluster1['auxIndex'][idx]
#                     break
#             S_i_info = {'Sequence': S_i, 'auxIndex': auxIndex_i}
#             for S_j in cluster2['Inst']:
#                 auxIndex_j = None
#                 for idx, sequence in enumerate(cluster2['Inst']):
#                     if np.array_equal(sequence, S_j):
#                         auxIndex_j = cluster2['auxIndex'][idx]
#                         break
#                 S_j_info = {'Sequence': S_j, 'auxIndex': auxIndex_j}
#                 if self.__IsOverlap(S_i_info, S_j_info):
#                     N += 1
#                     break  # Only need to find one overlap per S_i
#
#         min_len = min(len(cluster1['Inst']), len(cluster2['Inst']))
#         if N > epsilon * min_len:
#             mag_cluster1 = sum([self.__Magnitude(seq) for seq in cluster1['Inst']])
#             mag_cluster2 = sum([self.__Magnitude(seq) for seq in cluster2['Inst']])
#             if len(cluster1['Inst']) > len(cluster2['Inst']) or (
#                     len(cluster1['Inst']) == len(cluster2['Inst']) and mag_cluster1 > mag_cluster2):
#                 return True, False
#             else:
#                 return False, True
#
#         return True, True  # If overlap is not significant, keep both
#
#     def fit(self, time_series):
#         for i in range(len(time_series) - self.m):
#             self.__extract_subsequence(time_series, i)
#
#         self.Bm = self.__SubGroup(self.R, self.C, self.G)
#
#         # Prepare to test and handle overlapping clusters
#         keep_indices = set(range(len(self.Bm)))  # Initially, assume all clusters are to be kept
#
#         for i in range(len(self.Bm) - 1):
#             for j in range(i + 1, len(self.Bm)):
#                 if i in keep_indices and j in keep_indices:  # Process only if both clusters are still marked to keep
#                     keep_i, keep_j = self.__OLTest(self.Bm[i], self.Bm[j], self.epsilon)
#
#                     # Update keep indices based on OLTest outcome
#                     if not keep_i:
#                         keep_indices.remove(i)
#                     if not keep_j:
#                         keep_indices.remove(j)
#
#         # Filter self.routines to keep only those clusters marked for keeping
#         self.Bm = [self.Bm[i] for i in keep_indices]
#
#     def show_results(self):
#         print("Routines detected: ", len(self.Bm))
#         print("_" * 50)
#         for i, b in enumerate(self.Bm):
#             print(f"Centroid {i + 1}: {b['Cent']}")
#             print(f"Routine {i + 1}: {b['Inst']}")
#             print(f"Date {i + 1}: {b['Date']}")
#             print("\n", "-" * 50, "\n")
#
#     def get_results(self):
#         return self.Bm


if __name__ == "__main__":
    args = argparser.parse_args()
    df = load_data(args.data_dir)
    correspondencies = obtain_correspondencies(args.dictionary_dir)
    feat_extraction = feature_extraction(df, correspondencies)
    time_series = get_time_series(feat_extraction, "gym")
    routine_detector = DRFL(args.param_m, args.param_R, args.param_C, args.param_G, args.epsilon)
    # routine_detector = DRFL(7, 40, 5, 60, 0.2)
    routine_detector.fit(time_series)
    routine_detector.show_results()
    routines_detected = routine_detector.get_results()
    routine_detector.plot_results(title="Gym Routine", title_fontsize=40, labels_fontsize=35,
                                  xlim=(pd.to_datetime("2024-02-01"), pd.to_datetime("2024-03-31")),
                                  xlabel="Date", ylabel="Gym Frequency", ticks_fontsize=30,
                                  xticklabels_rotation=90, figsize=(40, 20), legend_title="Routines",
                                  legend_title_fontsize=40, legend_labels_fontsize=35, show_legend=True)



