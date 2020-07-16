from utils.set_seed import set_seed

set_seed(0)

import os
import json5
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%y%m%d-%H%M%S")

import training
from game.game import Game
from agent import q_agent
from utils.dataprep import load_emb_gz, make_categories

SETTINGS_FILE = "settings.json5"


def run_experiment(model_dir, load_file, save_file, mode,
                   dataset, trim_dataset_to_n_images, use_categories, vocabulary_size,
                   n_images_to_guess_from, **kwargs):
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
        "categories": categories,
        "reward_sender": {"success": 1, "fail": 0},
        "reward_receiver": {"success": 1, "fail": 0}
    }

    agent_args = {
        "input_shapes": [embeddings[0].shape] * n_images_to_guess_from + [(1,)],
        "output_size": vocabulary_size,
        "n_symbols": vocabulary_size,
        "embedding_size": 50,
        "learning_rate": 0.002,
        "use_bias": True,
        "loss": "binary_crossentropy",
        "optimizer": "adam"
    }

    filename = os.path.join(model_dir, save_file)
    filename = f"{filename}.json5"
    print(f"Writing parameters to '{filename}' ...")
    with open(filename, "w") as f:
        json5.dump(experiment_args, f, indent="    ")

    agent1 = q_agent.MultiAgent(**agent_args, role="sender")
    agent2 = q_agent.MultiAgent(**agent_args, role="receiver")

    if load_file:
        filename = os.path.join(model_dir, load_file)
        print(f"Loading weights from '{filename}' ...")
        agent1.load(f"{filename}.01")
        agent2.load(f"{filename}.02")

    game = Game(**game_args)

    if mode == "train":
        training.run_training(game, agent1, agent2, n_images_to_guess_from=n_images_to_guess_from,
                              **kwargs)
    elif mode == "test":
        raise NotImplementedError("test mode")
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    if save_file:
        filename = os.path.join(model_dir, save_file)
        print(f"Saving weights to '{filename}' ...")
        agent1.save(f"{filename}.01")
        agent2.save(f"{filename}.02")


if __name__ == "__main__":
    with open(SETTINGS_FILE, "r") as f:
        experiment_args = json5.load(f)
    run_experiment(**experiment_args)
