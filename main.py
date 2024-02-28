import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
def load_data(data_dir: str):
    data = pd.read_csv(data_dir, sep=",", header=0, index_col=False)
    return data


print("Loading data...")
data = load_data("data/activities-simulation.csv")
print("Data loaded successfully")
# print(data.head(10))
print(data["Sequence"].values)