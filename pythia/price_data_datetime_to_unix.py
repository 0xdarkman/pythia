import pandas as pd
import os
import numpy as np

DATA_FOLDER = "/home/bernhard/repos/pythia/data/recordings/poloniex/processed"

csvs = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

all_series = dict()
for csv in csvs:
    p = os.path.join(DATA_FOLDER, csv)
    s = pd.read_csv(p, index_col="timestamp", parse_dates=["timestamp"])
    all_series[csv[:-4]] = s


for n in all_series:
    s = all_series[n]
    s.index = s.index.astype(np.int64) // 10**9
    s.to_csv(os.path.join(DATA_FOLDER, (n + ".csv")))
