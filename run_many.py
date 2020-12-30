import pandas as pd
import sys, os
import yaml
import csv
from datetime import datetime

import run


QUEUE_FILE = "settings-reinforce-1.csv"
RESULTS_FILE = "settings-reinforce-1.results.csv"


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
        df = pd.read_csv(f, sep=";")
    for i in df.index:
        row = df.iloc[i, :]
        experiment_args = row.to_dict()
        print(experiment_args)
        for k in experiment_args:
            if type(experiment_args[k]).__module__ == "numpy":
                experiment_args[k] = experiment_args[k].item()
        with open("temp.yml", "w") as f:
            yaml.safe_dump(experiment_args, f, indent=4)
        # results_summary = run.run(**experiment_args)
        try:
            results_summary = run.run(**experiment_args)
        except Exception as e:
            print(e)
            results_summary = {"termination": repr(e)}
        for k, v in results_summary.items():
            row.loc[k] = v
        if not os.path.isfile(results_file):
            pd.DataFrame(row).transpose().to_csv(results_file, mode="a", header=True)
        else:
            pd.DataFrame(row).transpose().to_csv(results_file, mode="a", header=False)


if __name__ == "__main__":
    main()
