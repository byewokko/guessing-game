import pandas as pd
import sys, os
import yaml
from datetime import datetime

import run


SCHEDULE_FILE = "schedule.csv"
RESULTS_FILE = "results.csv"


def main():
    with open(SCHEDULE_FILE) as f:
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
            results = run.run_training(**experiment_args)
            for k, v in results.items():
                row[k] = v
            experiment_args["mode"] = "test"
            experiment_args["load_file"] = experiment_args["save_file"]
            filename = experiment_args["save_file"]
            param_filename = os.path.join(experiment_args["model_dir"], f"{filename}.yaml")
            print(f"Writing parameters to '{param_filename}' ...")
            with open(param_filename, "w") as f:
                yaml.safe_dump({str(k): str(v) for k, v in experiment_args.items()}, f, indent=4)
            if not os.path.isfile(RESULTS_FILE):
                pd.DataFrame(row).transpose().to_csv(RESULTS_FILE, mode="a", header=True)
            else:
                pd.DataFrame(row).transpose().to_csv(RESULTS_FILE, mode="a", header=False)
        else:
            raise ValueError(f"Invalid mode: '{mode}'")


if __name__ == "__main__":
    main()
