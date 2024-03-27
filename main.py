import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import cpu_count
from itertools import product

from src.DRFL import DRFL, ParallelSearchDRFL

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


if __name__ == "__main__":
    ts = [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]
    for x in range(4):
        ts += ts

    print(ts)

    time_series = pd.Series(np.array(ts))
    # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
    target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]

    R = [x for x in range(1, 6, 1)]
    C = [x for x in range(1, 10, 1)]
    G = [x for x in range(1, 6, 1)]
    epsilon = [0.5, 1]
    alpha = 0.5
    sigma = 3

    parallel_search = ParallelSearchDRFL(n_jobs=10, alpha=alpha, sigma=sigma, param_grid={'m': 3, 'R': R, 'C': C, 'G': G, 'epsilon': epsilon})
    parallel_search.fit(time_series, target_centroids=target_centroids)
    results = parallel_search.cv_results()
    best_params = parallel_search.best_params()
    print(results.head())
    print(best_params)
    best_drfl = DRFL(3, best_params["R"], best_params["C"], best_params["G"], best_params["epsilon"])
    best_drfl.fit(time_series)
    best_drfl.show_results()
    best_drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
                           yticks_fontsize=20, figsize=(45, 25),
                           linewidth_bars=2, xlim=(0,50))

    # drfl = DRFL(3, 2, 3, 4, 0.5)
    # drfl.fit(time_series)
    # drfl.show_results()
    # print(drfl.estimate_distance(target_centroids, 0.5, 0.5, 2))
    #

    # drfl = DRFL(3, 1.0, 2,  3.0, 0.5)
    # drfl.fit(time_series)
    # drfl.show_results()
    # drfl.plot_results(title_fontsize=40, labels_fontsize=35,
    #                   xlim=(0, 15), xticks_fontsize=18,
    #                   yticks_fontsize=20, figsize=(40, 20),
    #                   linewidth_bars=5)

    # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # PARALLEL SEARCH WITH COMPLICATED TIME SERIES

    # args = argparser.parse_args()
    # df = load_data(args.data_dir)
    # correspondencies = obtain_correspondencies(args.dictionary_dir)
    # feat_extraction = feature_extraction(df, correspondencies)
    # time_series = get_time_series(feat_extraction, "gym")
    # R_params = [x for x in range(20, 70, 10)]
    # C_params = [x for x in range(5, 9)]
    # G_params = [x for x in range(20, 100, 10)]
    # time_series = get_time_series(feat_extraction, "gym")
    # target_centroids = [[28, 0, 44, 0, 125, 11, 0], [79, 0, 47, 0, 66, 118, 0]]
    # params = list(product(R_params, C_params, G_params, [1]))
    # alpha, sigma = 0.6, 4
    #
    # # Sequential search
    # st = time.time()
    # result = []
    # for R, C, G, epsilon in params:
    #     drfl = DRFL(7, R, C, G, epsilon)
    #     drfl.fit(time_series)
    #     mean_distance = drfl.estimate_distance(target_centroids, alpha, sigma)
    #     result.append({"R": R, "C": C, "G": G, "epsilon": epsilon, "mean_distance": mean_distance})
    #
    # print(f"Elapsed sequential time: {time.time() - st}")
    #
    # param_grid = {'m': 7, 'R': R_params, 'C': C_params, 'G': G_params, 'epsilon': [1], 'alpha': [alpha],
    #               'sigma': [sigma]}
    #
    # # Parallel search: comparing time
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=1, param_grid=param_grid)
    # mDRFL.fit(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers=1: {time.time() - st}")
    #
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=5, param_grid=param_grid)
    # mDRFL.fit(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers=5: {time.time() - st}")
    #
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=10, param_grid=param_grid)
    # mDRFL.fit(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers=10: {time.time() - st}")
    #
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=cpu_count() - 2, param_grid=param_grid)
    # mDRFL.fit(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers={cpu_count() - 2}: {time.time() - st}")
    #
    # # Results
    # results = mDRFL.cv_results()
    # best_params = mDRFL.best_params()
    # top = results.head()
    # print(top)
    #
    # best_drfl = DRFL(7, best_params["R"], best_params["C"], best_params["G"], best_params["epsilon"])
    # best_drfl.fit(time_series)
    # best_drfl.show_results()
    # best_drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
    #                        yticks_fontsize=20, figsize=(45, 20),
    #                        linewidth_bars=2)
