import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os

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


def plot_feat_extraction(feat_extract: pd.DataFrame):
    columnas = feat_extract.columns[3:].tolist()
    for col in columnas:
        key = int(col.replace("N_", ""))
        plt.bar(feat_extract["Day"], feat_extract[col], label=correspondencies[key])
    plt.legend()
    plt.show()



if __name__ == "__main__":
    df = load_data(argparser.parse_args().data_dir)
    correspondencies = obtain_correspondencies(argparser.parse_args().dictionary_dir)
    feat_extraction = feature_extraction(df, correspondencies)
    for col in feat_extraction.columns[3:]:
        print(sum(feat_extraction[col] > 0), correspondencies[int(col.replace("N_", ""))])
    plot_feat_extraction(feat_extraction)

