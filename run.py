from utils.set_seed import set_seed

set_seed(0)

import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%y%m%d-%H%M%S")

import training
from game.game import Game
from agent import q_agent
from utils.dataprep import load_emb_gz, make_categories

experiment_args = {
    # Execution mode: "train" or "test"
    "mode": "train",

    # Directory to load from and save to
    "model_dir": "models",

    # Weights file to load: file prefix or None (to start with a blank model)
    "load_file": None,

    # Prefix used for saving weights, result and log files
    "save_file": f"{TIMESTAMP}-justtesting",

    # Roles mode: "switch" or "fixed"
    # TODO: add support for simple fixed-role agents
    "roles": "switch",

    # Number of episodes to play in total
    "n_episodes": 5000,

    # Number of episodes between updates
    "batch_size": 30,

    # Path to embedding file
    "dataset": "data/imagenet-4000-vgg19.emb.gz",
    # "dataset": "data/esp-10000-vgg19.emb.gz",
    # "dataset": "data/esp-10000-xception.emb.gz",

    # Number of images to play with (-None- to keep the whole dataset)
    "trim_dataset_to_n_images": None,

    # Pick images based on categories (generated from the first column in the embedding file)
    "use_categories": True,

    # Number of symbols that the agents are allowed to use
    "vocabulary_size": 50,

    # Size of the first layer in the agents' networks
    "embedding_size": 50,

    # Number of images shown presented in each turn of the game
    "n_images_to_guess_from": 2,

    # Sender type: "agnostic" or "informed"
    # TODO: add informed sender as a parameter
    "sender_type": "agnostic",
}


def run_experiment(model_dir, load_file, save_file, mode, roles,
                   dataset, trim_dataset_to_n_images, use_categories, vocabulary_size,
                   n_images_to_guess_from, **kwargs):
    print(f"Loading image embeddings from '{dataset}' ...")
    path2ind, path_list, embeddings = load_emb_gz(dataset,
                                                  trim_dataset_to_n_images)
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
        "loss": "mse",
        "optimizer": "adam"
    }

    filename = os.path.join(model_dir, save_file)
    filename = f"{filename}.par.txt"
    print(f"Writing parameters to '{filename}' ...")
    with open(filename, "w") as f:
        print(experiment_args, file=f)
        print(agent_args, file=f)
        print("", file=f)

    if roles == "switch":
        agent1 = q_agent.MultiAgent(**agent_args, role="sender")
        agent2 = q_agent.MultiAgent(**agent_args, role="receiver")
    elif roles == "fixed":
        agent1 = q_agent.Sender(**agent_args)
        # agent1 = q_agent.SenderInformed(agent_args)
        agent2 = q_agent.Receiver(**agent_args)
    else:
        raise ValueError(f"Unknown mode: '{roles}'")

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
    run_experiment(**experiment_args)
