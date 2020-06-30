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
    "model_dir": "models",
    "load_file": None,
    "save_file": f"{TIMESTAMP}-justtesting",
    "roles": "switch",
    "n_episodes": 20000,
    "batch_size": 30,
    "dataset": "data/imagenet-4000-vgg19.emb.gz",
    "trim_dataset_to_n_images": None,
    "use_categories": True,
    "vocabulary_size": 50,
    "embedding_size": 50,
    "n_images_to_guess_from": 2
}


def run_experiment(model_dir, load_file, save_file, roles,
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
        print(game_args, file=f)
        print("", file=f)

    if roles == "switch":
        agent1 = q_agent.MultiAgent(**agent_args)
        agent2 = q_agent.MultiAgent(**agent_args)
    elif roles == "fixed":
        agent1 = q_agent.Sender(**agent_args)
        # agent1 = q_agent.SenderInformed(agent_args)
        agent2 = q_agent.Receiver(**agent_args)
    else:
        raise ValueError(f"Unknown mode: '{roles}'")

    if load_file:
        filename = os.path.join(model_dir, load_file)
        agent1.load(f"{filename}.01")
        agent2.load(f"{filename}.02")

    game = Game(**game_args)

    training.run_training(game, agent1, agent2, **kwargs)

    if experiment_args["save_file"]:
        filename = os.path.join(model_dir, save_file)
        agent1.save(f"{filename}.01")
        agent2.save(f"{filename}.02")


if __name__ == "__main__":
    run_experiment(**experiment_args)
