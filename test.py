import pickle

from utils.set_seed import set_seed

import datetime
import os

import numpy as np
import yaml
import sys
from collections import OrderedDict

import pandas as pd

import tensorflow.keras.optimizers as optim
import tensorflow.keras


def create_agents(
		*,
		image_shape, number_of_images, embedding_size, vocabulary_size, sender_type,
		temperature, optimizer, algorithm, max_memory, exploration_decay, exploration_floor,
		role_mode, shared_embedding, role_alteration=False,
		**kwargs
):
	# SET UP AGENTS
	learning_rate = 0.1
	optimizers = {
		"adam": optim.Adam,
		"sgd": optim.SGD,
		"adadelta": optim.Adadelta,
		"rmsprop": optim.RMSprop
	}

	agent_settings = {
		"n_images": number_of_images,
		"input_image_shape": image_shape,
		"embedding_size": embedding_size,
		"vocabulary_size": vocabulary_size,
		"temperature": temperature,
		"optimizer": optimizers[optimizer](lr=learning_rate),
		"sender_type": sender_type,
		"max_memory": max_memory,
		"exploration_start": 0,
		"exploration_decay": exploration_decay,
		"exploration_floor": exploration_floor
	}

	tensorflow.keras.backend.clear_session()
	if algorithm == "reinforce":
		from agent.reinforce import Sender, Receiver, MultiAgent
	elif algorithm == "qlearning":
		from agent.qlearning import Sender, Receiver, MultiAgent
	else:
		raise ValueError(f"Expected 'reinforce' or 'qlearning' algorithm, got '{algorithm}'")

	if role_mode == "switch":
		sender = MultiAgent(
			active_role="sender",
			shared_embedding=shared_embedding,
			**agent_settings
		)
		receiver = MultiAgent(
			active_role="receiver",
			shared_embedding=shared_embedding,
			**agent_settings
		)
		if role_alteration:
			sender.switch_role()
			receiver.switch_role()
			sender, receiver = receiver, sender
	elif role_mode == "static":
		sender = Sender(**agent_settings)
		receiver = Receiver(**agent_settings)
	else:
		raise ValueError(f"Role mode must be either 'static' or 'switch', not '{role_mode}'")

	return sender, receiver


def run_one(
		agent1, agent2, game, testset,
		seed=None
):
	sender = agent1
	receiver = agent2
	role_setting = 0

	metrics = "episode role_setting images symbol guess success".split(" ")
	dtypes = [
		pd.Int32Dtype(), bool, object, pd.Int32Dtype(), pd.Int32Dtype(),
		pd.Float64Dtype()
	]
	test_log = pd.DataFrame(columns=metrics)
	for column, dtype in zip(metrics, dtypes):
		test_log[column] = test_log[column].astype(dtype)

	if seed is not None:
		set_seed(seed)

	episode = 0
	exit_status = "full"
	error = False

	batch_log = {metric: [] for metric in metrics}
	for test in testset:
		episode += 1
		game.reset()

		try:
			# Sender turn
			sender_ids = test["sender_ids"]
			sender_state = game.get_sender_state_from_ids(
				ids=sender_ids,
				expand=True
			)
			sender_probs = np.squeeze(sender.predict(
				state=sender_state
			))
			sender_action = sender.choose_action(sender_probs)

			# Receiver turn
			receiver_ids = test["receiver_ids"]
			receiver_pos = test["receiver_pos"]
			receiver_state = game.get_receiver_state_from_ids(
				receiver_ids,
				receiver_pos,
				sender_action,
				expand=True
			)
			receiver_probs = np.squeeze(receiver.predict(
				state=receiver_state
			))
			receiver_action = receiver.choose_action(receiver_probs)
		except Exception as e:
			print("\n", "ERROR", e)
			error = True
			break

		# Evaluate turn and remember
		sender_reward, receiver_reward, success = game.evaluate_guess(receiver_action)

		batch_log["episode"].append(episode)
		batch_log["role_setting"].append(role_setting)
		batch_log["images"].append(sender_ids)
		batch_log["symbol"].append(sender_action)
		batch_log["guess"].append(receiver_action)
		batch_log["success"].append(success)

		if not episode % 200:
			print(f"\r{episode} games played", end="")

	test_log = test_log.append(pd.DataFrame(batch_log))
	if error:
		return test_log, "error"

	print()

	return test_log, exit_status


def compute_final_stats(training_log, exit_status="full", analysis_window=None):
	if analysis_window:
		sample = training_log.tail(analysis_window)
	else:
		sample = training_log
	stats = {
		"exit_status": exit_status,
		"final_episode": training_log.iloc[-1]["episode"],
		"mean_success": sample["success"].mean()
	}
	frequent_symbols = sample["symbol"].value_counts(normalize=True)
	n_frequent_symbols = 0
	freq_sum = 0
	for freq in frequent_symbols:
		n_frequent_symbols += 1
		freq_sum += freq
		if freq_sum >= 0.9:
			break
	stats["n_frequent_symbols"] = n_frequent_symbols
	return stats


def run_many(test_path, dataset, model_path, model_folders, out_name, role_alteration, seed=None):
	stats_file = f"{out_name}.results.csv"
	# LOAD DATASET
	from utils.dataprep import load_emb_pickled
	metadata, embeddings = load_emb_pickled(dataset)
	filenames = metadata.get("fnames")
	categories = metadata.get("categories")
	image_shape = [len(embeddings[0])]

	# CREATE GAME
	game_settings = {
		"images": embeddings,
		"categories": categories,
		"images_filenames": filenames
	}
	from game import Game
	game = Game(**game_settings)

	# LOAD TEST
	with open(test_path, "rb") as f:
		test = pickle.load(f)

	for folder in model_folders:
		settings_path = os.path.join(model_path, folder, "settings.yml")
		print(f"Loading model from {settings_path}")
		with open(settings_path) as f:
			settings = yaml.safe_load(f)
		settings["image_shape"] = image_shape
		agent1, agent2 = create_agents(role_alteration=role_alteration, **settings)
		try:
			agent1.load(os.path.join(model_path, folder, "agent1"))
			agent2.load(os.path.join(model_path, folder, "agent2"))
		except Exception as e:
			print(f"Cannot load agents: {e}")
			continue

		print(f"Testing model {folder}")
		test_log, exit_status = run_one(agent1, agent2, game, test, seed)

		test_log_file = os.path.join(model_path, folder, f"{out_name}.csv")
		test_log.to_csv(test_log_file)
		print(f"Test log saved to {test_log_file}")

		stats = compute_final_stats(test_log, exit_status)
		# append stats to stats_file
		entry = OrderedDict()
		entry.update(settings)
		entry.update(stats)
		# create header if stats_file is not initzd
		if not os.path.isfile(stats_file):
			with open(stats_file, "w") as f:
				print(",".join(entry.keys()), file=f)
		with open(stats_file, "a") as f:
			print(",".join(map(str, entry.values())), file=f)
	print(f"Summary written to {stats_file}")


def main(config_file):
	with open(config_file, "r") as f:
		settings = yaml.load(f)
	if isinstance(settings["model_folders"], str):
		settings["model_folders"] = settings["model_folders"].strip().split("\n")
	run_many(**settings)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		config = sys.argv[1]
	else:
		config = "settings-test.yml"
	main(config)
