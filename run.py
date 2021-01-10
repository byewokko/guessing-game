import utils
from utils.set_seed import set_seed

set_seed(1)

import os
import yaml
import sys
from datetime import datetime

import training
from game.game import Game
from agent import q_agent, reinforce_agent
from utils.dataprep import load_emb_gz, make_categories

SETTINGS_FILE = "settings.yml"


def run(
        mode,
        model_dir,
        load_file,
        save_file,
        dataset,
        n_active_images,
        vocabulary_size,
        model_type,
        **kwargs):
    TIMESTAMP = datetime.now().strftime("%y%m%d-%H%M%S")
    save_file = save_file.format(TIMESTAMP=TIMESTAMP)
    try:
        metadata, features = utils.dataprep.load_emb_pickled(dataset)
        img_filenames = metadata.get("fnames")
        img_categories = metadata.get("categories")
    except FileNotFoundError:
        _, img_filenames, features = load_emb_gz(dataset)
        img_categories = make_categories(img_filenames)

    agent_args = {
        "input_shapes": [features[0].shape] * n_active_images + [(1,)],
        "vocabulary_size": vocabulary_size,
        "n_symbols": vocabulary_size,
        "n_images": n_active_images,
        "input_image_shape": features[0].shape
    }

    for k in ("embedding_size", "learning_rate", "gibbs_temperature", "loss", "optimizer",
              "explore", "model_type", "sender_settings", "receiver_settings", "shared_embedding"):
        if k in kwargs:
            agent_args[k] = kwargs[k]

    if model_type == "reinforce":
        # agent1 = reinforce_agent.MultiAgent(name="01", role="sender", **agent_args)
        # agent2 = reinforce_agent.MultiAgent(name="02", role="receiver", **agent_args)
        agent1 = reinforce_agent.Sender(name="01", **agent_args)
        agent2 = reinforce_agent.Receiver(name="02", **agent_args)
    else:
        agent1 = q_agent.MultiAgent(name="01", role="sender", **agent_args)
        agent2 = q_agent.MultiAgent(name="02", role="receiver", **agent_args)

    if load_file:
        load_filename = os.path.join(model_dir, load_file)
        print(f"Loading weights from '{load_filename}' ...")
        agent1.load(f"{load_filename}.01")
        agent2.load(f"{load_filename}.02")

    game_args = {
        "images": features,
        "images_filenames": img_filenames,
        "categories": img_categories
    }

    for k in ("reward_sender", "reward_receiver", "reward"):
        if k in kwargs:
            game_args[k] = kwargs[k]
    game = Game(**game_args)

    experiment_args = {
        "game": game,
        "agent1": agent1,
        "agent2": agent2,
        "n_active_images": n_active_images,
    }
    for k in ("n_episodes", "batch_size", "n_active_images", "roles", "show_plot", "early_stopping"):
        if k in kwargs:
            experiment_args[k] = kwargs[k]

    if mode == "test":
        assert load_file
        save_path = os.path.join(model_dir, save_file)
        res_file = f"{save_path}.res.csv"
        with open(res_file, "w") as rf:
            training.run_test(results_file=rf, **experiment_args)
    elif mode == "train":
        assert save_file
        save_path = os.path.join(model_dir, save_file)
        results = None
        results_summary, learning_curves = training.run_training(**experiment_args)
        if save_file:
            print(f"Saving weights to '{save_path}.*' ...")
            agent1.save(f"{save_path}.01")
            agent2.save(f"{save_path}.02")
            learning_curves.to_csv(f"{save_path}.curves.csv")
        print(results_summary)
        with open(f"temp.yml", "r") as f:
            experiment_args = yaml.safe_load(f)
        experiment_args["load_file"] = save_file
        experiment_args["mode"] = "test"
        param_filename = os.path.join(model_dir, f"{save_file}.yml")
        print(f"Writing parameters to '{param_filename}' ...")
        with open(param_filename, "w") as f:
            yaml.safe_dump(experiment_args, f, indent=4)
        return results_summary
    else:
        raise ValueError(f"Invalid mode: '{mode}'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = SETTINGS_FILE
    with open(settings_file, "r") as f:
        experiment_args = yaml.safe_load(f)
    with open("temp.yml", "w") as f:
        yaml.safe_dump(experiment_args, f, indent=4)
    run(**experiment_args)
