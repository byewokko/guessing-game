import pandas as pd
import sys, os
import yaml
from datetime import datetime

import run


QUEUE_FILE = "results/queue6.csv"
RESULTS_FILE = "results/results6.csv"


def main():
    if len(sys.argv) == 2:
        queue_file = sys.argv[1]
        if queue_file.endswith(".csv"):
            results_file = queue_file.replace(".csv", ".results.csv")
        else:
            results_file = f"{queue_file}.results.csv"
    elif len(sys.argv) > 2:
        queue_file = sys.argv[1]
        results_file = sys.argv[2]
    else:
        queue_file = QUEUE_FILE
        results_file = RESULTS_FILE

    with open(queue_file) as f:
        df = pd.read_csv(f)
    for i in df.index:
        row = df.iloc[i, :]
        experiment_args = dict(row)
        print(experiment_args)
        with open("temp.yml", "w") as f:
            yaml.safe_dump(experiment_args, f, indent=4)
        results_summary = run.run(**experiment_args)
        for k, v in results_summary.items():
            row.loc[k] = v
        if not os.path.isfile(results_file):
            pd.DataFrame(row).transpose().to_csv(results_file, mode="a", header=True)
        else:
            pd.DataFrame(row).transpose().to_csv(results_file, mode="a", header=False)


if __name__ == "__main__":
    main()
