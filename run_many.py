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
        mode = experiment_args["mode"]
        if mode == "test":
            assert experiment_args["load_file"]
            run.run_test(**experiment_args)
        elif mode == "train":
            assert experiment_args["save_file"]
            if "{TIMESTAMP}" in experiment_args["save_file"]:
                timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
                experiment_args["save_file"] = experiment_args["save_file"].format(TIMESTAMP=timestamp,
                                                                                   **experiment_args)
            row.loc["save_file"] = experiment_args["save_file"]
            print(experiment_args["save_file"])
            results = run.run_training(**experiment_args)
            for k, v in results.items():
                row.loc[k] = v
            experiment_args["mode"] = "test"
            experiment_args["load_file"] = experiment_args["save_file"]
            filename = experiment_args["save_file"]
            param_filename = os.path.join(experiment_args["model_dir"], f"{filename}.yaml")
            print(f"Writing parameters to '{param_filename}' ...")
            with open(param_filename, "w") as f:
                yaml.safe_dump({str(k): str(v) for k, v in experiment_args.items()}, f, indent=4)
            if not os.path.isfile(results_file):
                pd.DataFrame(row).transpose().to_csv(results_file, mode="a", header=True)
            else:
                pd.DataFrame(row).transpose().to_csv(results_file, mode="a", header=False)
        else:
            raise ValueError(f"Invalid mode: '{mode}'")


if __name__ == "__main__":
    main()
