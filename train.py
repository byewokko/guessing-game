import datetime
import os

import numpy as np
import yaml
import sys
from collections import OrderedDict

import pandas as pd

import tensorflow.keras.optimizers as optim
import tensorflow.keras


class EarlyStopping:
	def __init__(self, patience, min_episodes=0):
		self.patience = patience
		self.min_episodes = min_episodes
		self.max_score = None
		self.max_score_ep = None

	def check(self, episode, score):
		if episode < self.min_episodes:
			return False
		if self.max_score is None or score > self.max_score:
			self.max_score_ep = episode
			self.max_score = score
			return False
		if episode > self.max_score_ep + self.patience:
			return True


def run_one(
		*,
		out_dir, dataset, number_of_images, embedding_size, vocabulary_size, sender_type,
		temperature, number_of_episodes, batch_size, analysis_window, optimizer,
		memory_sampling_mode, algorithm, max_memory,
		exploration_start, exploration_decay, exploration_floor,
		**kwargs
):
	# TODO: refactor into settings parser
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

	# SET UP AGENTS
	learning_rate = 0.1
	optimizers = {
		"adam": optim.Adam,
		"sgd": optim.SGD
	}

	sender_settings = {
		"n_images": number_of_images,
		"input_image_shape": image_shape,
		"embedding_size": embedding_size,
		"vocabulary_size": vocabulary_size,
		"temperature": temperature,
		"optimizer": optimizers[optimizer](learning_rate),
		"sender_type": sender_type,
		#     "sender_type": "informed",
		#     "n_informed_filters": 20,
		"max_memory": max_memory,
		"exploration_start": exploration_start,
		"exploration_decay": exploration_decay,
		"exploration_floor": exploration_floor
	}
	receiver_settings = {
		"n_images": number_of_images,
		"input_image_shape": image_shape,
		"embedding_size": embedding_size,
		"vocabulary_size": vocabulary_size,
		"temperature": temperature,
		"optimizer": optimizers[optimizer](learning_rate),
		"max_memory": max_memory,
		"exploration_start": exploration_start,
		"exploration_decay": exploration_decay,
		"exploration_floor": exploration_floor,
	}

	tensorflow.keras.backend.clear_session()
	if algorithm == "reinforce":
		from agent.reinforce import Sender, Receiver
		sender = Sender(**sender_settings)
		receiver = Receiver(**receiver_settings)
	elif algorithm == "qlearning":
		from agent.qlearning import Sender, Receiver
		sender = Sender(**sender_settings)
		receiver = Receiver(**receiver_settings)

	metrics = "episode images symbol guess success sender_loss receiver_loss".split(" ")
	dtypes = [
		pd.Int32Dtype(), object, pd.Int32Dtype(), pd.Int32Dtype(),
		pd.Float64Dtype(), pd.Float64Dtype(), pd.Float64Dtype()
	]
	training_log = pd.DataFrame(columns=metrics)
	for column, dtype in zip(metrics, dtypes):
		training_log[column] = training_log[column].astype(dtype)

	episode = 0
	early_stopping = EarlyStopping(patience=5000, min_episodes=10000)
	batch_log = {metric: [] for metric in metrics}
	while episode < number_of_episodes:
		batch_log = {metric: [] for metric in metrics}
		while True:
			episode += 1
			game.reset()

			# Sender turn
			sender_state, img_ids = game.get_sender_state(
				n_images=number_of_images,
				unique_categories=True,
				expand=True,
				return_ids=True
			)
			sender_probs = np.squeeze(sender.predict(
				state=sender_state
			))
			sender_action = sender.choose_action(sender_probs)

			# Receiver turn
			receiver_state = game.get_receiver_state(
				sender_action,
				expand=True
			)
			receiver_probs = np.squeeze(receiver.predict(
				state=receiver_state
			))
			receiver_action = receiver.choose_action(receiver_probs)

			# Evaluate turn and remember
			sender_reward, receiver_reward, success = game.evaluate_guess(receiver_action)
			sender.remember(
				state=sender_state,
				action=np.asarray([sender_action]),
				action_probs=sender_probs,
				reward=np.asarray([sender_reward])
			)
			# print(np.asarray([sender_action]), np.asarray([receiver_action]), np.asarray([sender_reward]))
			receiver.remember(
				state=receiver_state,
				action=np.asarray([receiver_action]),
				action_probs=receiver_probs,
				reward=np.asarray([receiver_reward])
			)

			batch_log["episode"].append(episode)
			batch_log["images"].append(img_ids)
			batch_log["symbol"].append(sender_action)
			batch_log["guess"].append(receiver_action)
			batch_log["success"].append(success)

			if not episode % 500:
				stats = compute_live_stats(
					training_log=training_log,
					analysis_window=500,
					overwrite_line=False
				)

			if episode % batch_size == 0:
				break

		# Train on batch
		batch_log["sender_loss"] = sender.update_on_batch(batch_size, memory_sampling_mode=memory_sampling_mode)
		batch_log["receiver_loss"] = receiver.update_on_batch(batch_size, memory_sampling_mode=memory_sampling_mode)
		training_log = training_log.append(pd.DataFrame(batch_log))

		stats = compute_live_stats(
			training_log=training_log,
			analysis_window=analysis_window
		)

		if early_stopping.check(episode, stats["mean_success"]):
			break
	sender.save(os.path.join(out_dir, "sender.weights"))
	receiver.save(os.path.join(out_dir, "receiver.weights"))

	return training_log


def compute_final_stats(training_log, analysis_window=None):
	if not analysis_window:
		analysis_window = 30
	final_episode = training_log.iloc[-1]["episode"]
	tail = training_log.tail(analysis_window)
	stats = {
		"final_episode": training_log.iloc[-1]["episode"],
		"mean_success": tail["success"].mean()
	}
	frequent_symbols = tail["symbol"].value_counts(normalize=True)
	n_frequent_symbols = 0
	freq_sum = 0
	for freq in frequent_symbols:
		n_frequent_symbols += 1
		freq_sum += freq
		if freq_sum >= 0.9:
			break
	stats["n_frequent_symbols"] = n_frequent_symbols
	return stats


def compute_live_stats(training_log: pd.DataFrame, analysis_window, overwrite_line=True):
	LIVE_STATS_MSG = "\rEP{episode:05d}: \
	success {success:.3f}, \
	freq symbols {n_frequent_symbols:3d}, \
	sender loss: {sender_loss:.3f}, \
	receiver loss: {receiver_loss:.3f}".replace("\t", "")
	tail = training_log.tail(analysis_window)
	episode = tail.iloc[-1]["episode"]
	stats = {
		"mean_success": tail["success"].mean(),
		"mean_sender_loss": tail["sender_loss"].mean(),
		"mean_receiver_loss": tail["receiver_loss"].mean()
	}
	frequent_symbols = tail["symbol"].value_counts(normalize=True)
	n_frequent_symbols = 0
	freq_sum = 0
	for freq in frequent_symbols:
		n_frequent_symbols += 1
		freq_sum += freq
		if freq_sum >= 0.9:
			break
	stats["n_frequent_symbols"] = n_frequent_symbols
	print(LIVE_STATS_MSG.format(
		episode=episode,
		success=stats["mean_success"],
		n_frequent_symbols=stats["n_frequent_symbols"],
		sender_loss=stats["mean_sender_loss"],
		receiver_loss=stats["mean_receiver_loss"]
	), end="")
	if not overwrite_line:
		print()
	return stats


def run_many(settings_list, name):
	stats_file = f"{name}.stats.csv"
	for settings in settings_list:
		timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
		folder = os.path.join("models", f"{name}-{timestamp}")
		os.makedirs(folder)
		settings["out_dir"] = folder
		training_log: pd.DataFrame = run_one(**settings)
		# save training_data to training_data_file
		training_log_file = os.path.join(folder, "training_log.csv")
		training_log.to_csv(training_log_file)
		# compute stats
		stats = compute_final_stats(training_log)
		# append stats to stats_file
		entry = OrderedDict()
		entry.update(settings)
		entry.update(stats)
		# create header if stats_file is not initzd
		if not os.path.isfile(stats_file):
			with open(stats_file, "w") as f:
				print(",".join(entry.keys()), file=f)
		with open(stats_file, "a") as f:
			print(",".join(entry.values()), file=f)


def main(filename):
	if filename.endswith(".csv"):
		# RUN MANY
		# parse csv into a list of settings-dicts
		import messytables
		with open(filename, "rb") as f:
			row_set = messytables.CSVRowSet("", f)
			offset, headers = messytables.headers_guess(row_set.sample)
			row_set.register_processor(messytables.headers_processor(headers))
			row_set.register_processor(messytables.offset_processor(offset + 1))
			types = messytables.type_guess(row_set.sample, strict=True)
			row_set.register_processor(messytables.types_processor(types))
			settings_list = row_set.dicts()
		name = filename.replace(".csv", "")
		run_many(settings_list, name)
	else:
		# RUN ONE
		# parse yaml into a settings-dict
		with open(filename, "r") as f:
			settings = yaml.load(f)
		training_log = run_one(**settings)
		training_log_file = os.path.join(settings["out_dir"], "training_log.csv")
		training_log.to_csv(training_log_file)
		stats = compute_final_stats(training_log)
		training_stats_file = os.path.join(settings["out_dir"], "training_stats.yml")
		with open(training_stats_file, "w") as f:
			yaml.dump(stats, f)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		# filename = "settings-reinforce-1.csv"
		filename = "settings-new.yml"
	main(filename)
