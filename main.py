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


if __name__ == "__main__":
    args = argparser.parse_args()
    df = load_data(args.data_dir)
    correspondencies = obtain_correspondencies(args.dictionary_dir)
    feat_extraction = feature_extraction(df, correspondencies)
    time_series = get_time_series(feat_extraction, "gym")
    routine_detector = DRFL(args.param_m, args.param_R, args.param_C, args.param_G, args.epsilon)
    routine_detector.fit(time_series)
    routine_detector.show_results()
    routines_detected = routine_detector.get_results()
    routine_detector.plot_results(title_fontsize=40, labels_fontsize=35,
                                  xlim=(0, 100), xticks_fontsize=18,
                                  yticks_fontsize=20, figsize=(40, 20),
                                  linewidth_bars=5)



