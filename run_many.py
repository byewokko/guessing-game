import pandas as pd
import sys, os

import run


SCHEDULE_FILE = "schedule.csv"
RESULTS_FILE = "results.csv"


def main():
    with open(SCHEDULE_FILE) as f:
        df = pd.read_csv(f)
    for i in df.index:
        params = dict(df.iloc[i, :])
        results = run.run_training(**params)
        for k, v in results.items():
            params[k] = v
        if not os.path.isfile(RESULTS_FILE):
            df.to_csv(RESULTS_FILE, mode="a", header=True)
        else:
            df.to_csv(RESULTS_FILE, mode="a", header=False)


if __name__ == "__main__":
    main()
