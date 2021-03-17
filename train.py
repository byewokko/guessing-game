import datetime
import os
import yaml
import sys
from collections import OrderedDict

import pandas as pd

import tensorflow.keras.optimizers as optim
import tensorflow.keras



def dummy():
	if True or list(map(int, keras.__version__.split("."))) < [2, 4, 0]:
		import tensorflow.keras.layers as layers
		import tensorflow.keras.backend as K
		import tensorflow.keras.optimizers as optim
		from tensorflow.keras.models import Model, Sequential
	else:
		import keras.layers as layers
		import keras.backend as K
		import keras.optimizers as optim
		from keras.models import Model, Sequential

	import matplotlib.pyplot as plt
	import numpy as np
	import tensorflow as tf

	from game.game import Game



def run_one(settings):
	# TODO: refactor into settings parser
	# LOAD DATASET
	from utils.dataprep import load_emb_pickled
	DATASET = settings.get("dataset")
	metadata, embeddings = load_emb_pickled(DATASET)
	filenames = metadata.get("fnames")
	categories = metadata.get("categories")
	IMAGE_SHAPE = [len(embeddings[0])]

	N_IMAGES = settings.get("number_of_images")
	EMBEDDING_SIZE = settings.get("embedding_size")
	VOCABULARY_SIZE = settings.get("vocabulary_size")

	TEMPERATURE = settings.get("temperature")
	N_EPISODES = settings.get("number_of_episodes")
	ANALYSIS_WINDOW = settings.get("analysis_window")
	BATCH_SIZE = settings.get("batch_size")

	LEARNING_RATE = 0.1
	OPTIMIZER = settings.get("optimizer")
	optimizers = {
		"adam": optim.Adam,
		"sgd": optim.SGD
	}

	ADAPTIVE_TEMPERATURE = 0

	SENDER_KWARGS = {
		"n_images": N_IMAGES,
		"input_image_shape": IMAGE_SHAPE,
		"embedding_size": EMBEDDING_SIZE,
		"vocabulary_size": VOCABULARY_SIZE,
		"temperature": TEMPERATURE,
		"optimizer": optimizers[OPTIMIZER](LEARNING_RATE),
		"sender_type": "agnostic",
		#     "sender_type": "informed",
		#     "n_informed_filters": 20,
		"verbose": True
	}

	RECEIVER_KWARGS = {
		"n_images": N_IMAGES,
		"input_image_shape": IMAGE_SHAPE,
		"embedding_size": EMBEDDING_SIZE,
		"vocabulary_size": VOCABULARY_SIZE,
		"temperature": TEMPERATURE,
		"optimizer": optimizers[OPTIMIZER](LEARNING_RATE),
	}

	training_data = None
	return training_data


def compute_stats(training_data):
	stats = OrderedDict()
	return stats


def run_many(settings_list, name):
	stats_file = f"{name}.stats.csv"
	for settings in settings_list:
		timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
		folder = os.path.join("models", f"{name}-{timestamp}")
		os.makedirs(folder)
		settings["out_dir"] = folder
		training_data: pd.DataFrame = run_one(settings)
		# save training_data to training_data_file
		training_data_file = os.path.join(folder, "training_data.csv")
		training_data.to_csv(training_data_file)
		# compute stats
		stats = compute_stats(training_data)
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
		training_data = run_one(settings)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		filename = "settings-reinforce-1.csv"
	main(filename)
