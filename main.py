import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
from itertools import combinations
from collections import defaultdict

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", type=str, default="data/activities-simulation.csv", help="Path to the data file")
argparser.add_argument("--dictionary_dir", type=str, default="data/dictionary_rooms.json",
                       help="Path to the dictionary file")


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


#
# def Mag(S):
#     # Placeholder for the magnitude function
#     return np.max(S)
#
#
# def Dist(S1, S2):
#     # Placeholder for the distance function
#     return np.max(np.abs(S1 - S2))
#
#
# def NTM(Si, Sj, R):
#     # returns True if the distance between Si and Sj is less than or equal to R
#     return Dist(Si, Sj) <= R
#
#
# def IsOverlap(Sm_i, Sn_j, i, j):
#     p = len(Sm_i)
#     q = len(Sn_j)
#     return ((i + p) < j) or ((j + q) < i)
#
#
# def OLTest(Sm, Sn, epsilon):
#     Km = True
#     Kn = True
#     N = 0
#     i = 1
#     j = 2
#
#     while i <= len(Sm) and j <= len(Sn):
#         while i <= len(Sm) and j <= len(Sn):
#             if IsOverlap(Sm[i - 1], Sn[j - 2], i - 1, j - 2) or IsOverlap(Sm[i - 1], Sn[j - 1], i - 1, j - 1):
#                 N += 1
#             j += 1
#         i += 1
#
#     if N > epsilon * min(len(Sm), len(Sn)):
#         if len(Sm) > len(Sn):
#             Km = True
#             Kn = False
#         elif len(Sm) < len(Sn):
#             Km = False
#             Kn = True
#         else:
#             if Mag(Sm) > Mag(Sn):
#                 Km = True
#                 Kn = False
#             else:
#                 Km = False
#                 Kn = True
#
#     return Km, Kn
#
#
# def SubGroup(S, R, C, G):
#     B = [{'Cent': S[0], 'Inst': [S[0]]}]
#     for Si in S:
#         if Mag(Si) > G:
#             distances = [Dist(Si, Bj['Cent']) for Bj in B]
#             j_hat = np.argmin(distances)
#             if not any(NTM(Si, Bj['Inst'], R) for Bj in B):
#                 B[j_hat]['Inst'].append(Si)
#                 # Update the center here
#             else:
#                 B.append({'Cent': Si, 'Inst': [Si]})
#         else:
#             B.append({'Cent': Si, 'Inst': [Si]})
#     B = [Bj for Bj in B if len(Bj['Inst']) >= C]
#     return B
#
#
# def extract_subsequences(time_series, m):
#     subsequences = []
#     for i in range(len(time_series) - m + 1):
#         subsequences.append(np.array(time_series[i:i + m]))
#     return subsequences
#
# def DRFL(time_series, m, R, C, G, epsilon):
#     Sm = extract_subsequences(time_series, m)
#     Bm = SubGroup(Sm, R, C, G)
#     for i in range(len(Bm) - 1):
#         for j in range(i, len(Bm)):
#             try:
#                 Ki, Kj = OLTest(Bm[i]['Inst'], Bm[j]['Inst'], epsilon)
#             except:
#                 print("ERROR", i, j, Bm, len(Bm))
#             if not Ki:
#                 Bm[i] = None
#     Bm = [b for b in Bm if b is not None]
#     return Bm


class DRFL:
    def __init__(self, m, R, C, G, epsilon):
        self.m = m
        self.R = R
        self.C = C
        self.G = G
        self.epsilon = epsilon
        self.Bm = None

    def fit(self, time_series):
        Sm, dates = self.__extract_subsequences(time_series, self.m)
        Bm = self.__SubGroup(Sm, self.R, self.C, self.G, dates)
        self.Bm = [b for b in Bm if b is not None]

    def show_results(self):
        if self.Bm is None:
            print("No routines detected or fit() not called")
            return
        print("N RUTINAS: ", len(self.Bm))
        for i, b in enumerate(self.Bm):
            print(f"CENTROIDE {i + 1}: {b['Cent']}")
            print(f"RUTINA {i + 1}: {b['Inst']}")
            print(f"FECHAS {i + 1}: {b['Date']}")
            print("_" * 50)

    @staticmethod
    def __Mag(S):
        return np.max(S)

    @staticmethod
    def __Dist(S1, S2):
        return np.max(np.abs(S1 - S2))

    def __NTM(self, Si, Sj, R):
        return self.__Dist(Si, Sj) <= R

    @staticmethod
    def __IsOverlap(Sm_i, Sn_j, i, j):
        p = len(Sm_i)
        q = len(Sn_j)
        return ((i + p) > j) and ((j + q) > i)

    def __OLTest(self, Sm, Sn, epsilon):
        N = 0
        for i in range(len(Sm)):
            for j in range(len(Sn)):
                if self.__IsOverlap(Sm[i], Sn[j], i, j):
                    N += 1
        Km, Kn = self.__decide_Km_Kn(len(Sm), len(Sn), self.__Mag(Sm), self.__Mag(Sn), N, epsilon)
        return Km, Kn

    def __SubGroup(self, S, R, C, G, dates):
        B = [{'Cent': S[0], 'Inst': [S[0]], 'Date': [dates[0]]}]
        for Si, date in zip(S[1:], dates[1:]):
            if self.__Mag(Si) > G:
                distances = [self.__Dist(Si, Bj['Cent']) for Bj in B]
                j_hat = np.argmin(distances)
                if not any(self.__NTM(Si, Bj['Inst'], R) for Bj in B):
                    B.append({'Cent': Si, 'Inst': [Si], 'Date': [date]})
                else:
                    B[j_hat]['Inst'].append(Si)
                    B[j_hat]['Date'].append(date)
            else:
                B.append({'Cent': Si, 'Inst': [Si], 'Date': [date]})
        B = [Bj for Bj in B if len(Bj['Inst']) >= C]
        return B

    @staticmethod
    def __extract_subsequences(time_series, m):
        subsequences = []
        dates = []
        for i in range(len(time_series) - m + 1):
            subsequences.append(np.array(time_series[i:i + m]))
            dates.append(time_series.index[i])
        return subsequences, dates

    @staticmethod
    def __decide_Km_Kn(len_Sm, len_Sn, Mag_Sm, Mag_Sn, N, epsilon):
        if N > epsilon * min(len_Sm, len_Sn):
            if len_Sm > len_Sn:
                return True, False
            elif len_Sm < len_Sn:
                return False, True
            else:
                return (Mag_Sm > Mag_Sn), (Mag_Sm <= Mag_Sn)
        return True, True  # Default to True, True if the condition doesn't meet


if __name__ == "__main__":
    df = load_data(argparser.parse_args().data_dir)
    df = load_data("data/activities-simulation-medium.csv")
    correspondencies = obtain_correspondencies(argparser.parse_args().dictionary_dir)
    feat_extraction = feature_extraction(df, correspondencies)
    # plot_gym_hours(feat_extraction)
    time_series = get_time_series(feat_extraction, "gym")
    routine_detector = DRFL(4, 10, 4, 60, 0.8)
    routine_detector.fit(time_series)
    routine_detector.show_results()
    # # Example time series data and parameters (placeholders, adjust according to actual data and requirements)
    # Bm = DRFL(time_series, 5, 2, 5, 80, 0.5)
    # print("N RUTINAS: ", len(Bm))
    # print("RUTINAS: ")
    # for i in range(len(Bm)):
    #     print("CENTROIDE ", i + 1, ":", Bm[i]['Cent'])
    #     print("RUTINA ", i + 1, ":", Bm[i]['Inst'])
    #     print("_" * 50)
