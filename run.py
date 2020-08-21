from utils.set_seed import set_seed

set_seed(2)

import os
import yaml
import sys
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%y%m%d-%H%M%S")

import training
from game.game import Game
from agent import q_agent
from utils.dataprep import load_emb_gz, make_categories

SETTINGS_FILE = "settings.yaml"


def run_training(model_dir, load_file, save_file, dataset, use_categories,
                 vocabulary_size, n_active_images, trim_dataset_to_n_images=False, **experiment_args):
    save_file = save_file.format(TIMESTAMP=TIMESTAMP)
    print(f"Loading image embeddings from '{dataset}' ...")
    path2ind, path_list, embeddings = load_emb_gz(dataset, trim_dataset_to_n_images)
    if use_categories:
        categories = make_categories(path_list)
    else:
        categories = None

    game_args = {
        "images": embeddings,
        "images_filenames": path_list,
        "categories": categories
    }

    for k in ("reward_sender", "reward_receiver", "reward"):
        if k in experiment_args:
            game_args[k] = experiment_args[k]

    agent_args = {
        "input_shapes": [embeddings[0].shape] * n_active_images + [(1,)],
        "output_size": vocabulary_size,
        "n_symbols": vocabulary_size,
    }

    for k in ("sender_type", "n_informed_filters", "embedding_size", "learning_rate", "gibbs_temperature",
              "loss", "optimizer", "use_bias", "explore", "batch_mode", "memory_sampling_distribution", "dropout",
              "shared_embedding", "out_activation"):
        if k in experiment_args:
            agent_args[k] = experiment_args[k]

    filename = os.path.join(model_dir, save_file)

    agent1 = q_agent.MultiAgent(name="01", role="sender", **agent_args)
    agent2 = q_agent.MultiAgent(name="02", role="receiver", **agent_args)

    if load_file:
        print(f"Loading weights from '{filename}' ...")
        agent1.load(f"{filename}.01")
        agent2.load(f"{filename}.02")

    game = Game(**game_args)

    results = None

    try:
        results = training.run_training(game, agent1, agent2, n_active_images=n_active_images,
                                        **experiment_args)
    except (InterruptedError, KeyboardInterrupt):
        filename = f"{filename}-interrupted"
    finally:
        if save_file:
            print(f"Saving weights to '{filename}.*' ...")
            agent1.save(f"{filename}.01")
            agent2.save(f"{filename}.02")

    print(results)
    return results


def run_test(model_dir, load_file, save_file, dataset, use_categories,
             vocabulary_size, n_active_images, trim_dataset_to_n_images=False, **experiment_args):
    save_file = save_file.format(TIMESTAMP=TIMESTAMP)
    print(f"Loading image embeddings from '{dataset}' ...")
    path2ind, path_list, embeddings = load_emb_gz(dataset, trim_dataset_to_n_images)
    if use_categories:
        categories = make_categories(path_list)
    else:
        categories = None

    game_args = {
        "images": embeddings,
        "images_filenames": path_list,
        "categories": categories
    }

    for k in ("reward_sender", "reward_receiver", "reward"):
        if k in experiment_args:
            game_args[k] = experiment_args[k]

    agent_args = {
        "input_shapes": [embeddings[0].shape] * n_active_images + [(1,)],
        "output_size": vocabulary_size,
        "n_symbols": vocabulary_size,
    }

    for k in ("sender_type", "n_informed_filters", "embedding_size", "learning_rate", "gibbs_temperature",
              "loss", "optimizer", "use_bias", "explore"):
        if k in experiment_args:
            agent_args[k] = experiment_args[k]

    agent_args["explore"] = None

    agent1 = q_agent.MultiAgent(name="01", role="sender", **agent_args)
    agent2 = q_agent.MultiAgent(name="02", role="receiver", **agent_args)

    if load_file:
        load_filename = os.path.join(model_dir, load_file)
        print(f"Loading weights from '{load_filename}' ...")
        agent1.load(f"{load_filename}.01")
        agent2.load(f"{load_filename}.02")
    else:
        raise RuntimeError("A load_file is required in test mode")

    game = Game(**game_args)

    save_filename = os.path.join(model_dir, save_file)
    res_file = f"{save_filename}.res.csv"
    with open(res_file, "w") as rf:
        training.run_test(game, agent1, agent2, rf, n_active_images=n_active_images,
                          **experiment_args)


def main(experiment_args):
    mode = experiment_args["mode"]
    if mode == "test":
        assert experiment_args["load_file"]
        run_test(**experiment_args)
    elif mode == "train":
        assert experiment_args["save_file"]
        if "{TIMESTAMP}" in experiment_args["save_file"]:
            experiment_args["save_file"] = experiment_args["save_file"].format(TIMESTAMP=TIMESTAMP)
        run_training(**experiment_args)
        experiment_args["mode"] = "test"
        experiment_args["load_file"] = experiment_args["save_file"]
        filename = experiment_args["save_file"]
        param_filename = os.path.join(experiment_args["model_dir"], f"{filename}.yaml")
        print(f"Writing parameters to '{param_filename}' ...")
        with open(param_filename, "w") as f:
            yaml.safe_dump(experiment_args, f, indent=4)
    else:
        raise ValueError(f"Invalid mode: '{mode}'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = SETTINGS_FILE
    with open(settings_file, "r") as f:
        experiment_args = yaml.safe_load(f)
    main(experiment_args)
