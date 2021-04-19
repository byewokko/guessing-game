import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import backend as K

from .agent import Agent, getshape

L = logging.getLogger(__name__)


def build_sender_model(
		n_images, input_image_shape, embedding_size,
		vocabulary_size, optimizer,
		sender_type="agnostic", image_embedding_layer=None,
		verbose=False, **kwargs
):
	image_inputs = [
		layers.Input(
			shape=input_image_shape,
			name=f"S_image_in_{i}",
			dtype="float32")
		for i in range(n_images)
	]

	if not image_embedding_layer:
		image_embedding_layer = layers.Dense(
			embedding_size,
			name="S_image_embedding"
		)

	# agnostic part
	sigmoid = layers.Activation("sigmoid", name="S_sigmoid")
	output_layer = layers.Dense(vocabulary_size, name="S_output")

	# informed part
	stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="S_stack")
	permute = layers.Permute([2, 1], name="S_permute")
	feature_filters = layers.Conv1D(
		filters=vocabulary_size,
		kernel_size=(1,),
		input_shape=[n_images, embedding_size],
		activation="sigmoid",
		data_format="channels_last",
		name="S_feature_filters"
	)

	permute_2 = layers.Permute([2, 1], name="S_permute_2")
	vocabulary_filter = layers.Conv1D(
		1,
		kernel_size=(1,),
		data_format="channels_last",
		name="S_vocabulary_filter"
	)
	flatten = layers.Flatten()

	output_activation = layers.Softmax()

	y = [image_embedding_layer(x) for x in image_inputs]
	if sender_type == "agnostic":
		y = [sigmoid(x) for x in y]
		y = layers.concatenate(y, axis=-1)
		y = output_layer(y)
	elif sender_type == "informed":
		y = stack(y)
		y = permute(y)
		y = feature_filters(y)
		y = permute_2(y)
		y = vocabulary_filter(y)
		y = flatten(y)

	y = output_activation(y)

	model_predict = models.Model([*image_inputs], y, name="S_predict")
	model_predict.compile(loss=losses.categorical_crossentropy, optimizer=optimizer)

	return model_predict


def build_receiver_model(
		n_images, input_image_shape, embedding_size,
		vocabulary_size, optimizer, image_embedding_layer=None, verbose=False, **kwargs
):
	image_inputs = [
		layers.Input(shape=input_image_shape, name=f"R_image_in_{i}", dtype="float32")
		for i
		in range(n_images)
	]

	if not image_embedding_layer:
		image_embedding_layer = layers.Dense(embedding_size, name="R_image_embedding")

	output_activation = layers.Softmax()

	symbol_input = layers.Input(shape=[1], dtype="int32", name=f"R_symbol_in")
	symbol_embedding = layers.Embedding(
		input_dim=vocabulary_size,
		output_dim=embedding_size,
		name="R_symbol_embedding"
	)
	dot_product = layers.Dot(axes=-1, name="R_dot_product")

	# connect the pipeline
	y_images = [image_embedding_layer(x) for x in image_inputs]
	y_symbol = symbol_embedding(symbol_input)
	y = [dot_product([img, y_symbol]) for img in y_images]
	y = layers.concatenate(y, axis=-1)
	y = output_activation(y)

	model_predict = models.Model([*image_inputs, symbol_input], y, name="R_predict")
	model_predict.compile(loss=losses.categorical_crossentropy, optimizer=optimizer)

	return model_predict


class QAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.max_memory = 2000
		self.memory_sampling_dist = None

	def predict(self, state):
		x = state
		return self.model.predict_on_batch(x)

	def update(self, state, action, target):
		x = [*state, action]
		return self.model_train.train_on_batch(x=x, y=target)

	def remember(self, state, action, target):
		x = [*state, action]
		self.memory_x.append(x)
		self.memory_y.append(target)

	def reset_memory(self):
		self.memory_x = []
		self.memory_y = []

	def update_on_batch(self, batch_size: int, reset_after=True, **kwargs):
		loss = []
		for x, y in zip(*self.make_batch(batch_size, kwargs.get("memory_sampling_mode"))):
			loss.append(self.model_train.train_on_batch(x=x, y=y))
		if reset_after:
			self.reset_memory()
		return loss

	def make_batch(self, batch_size: int, memory_sampling_mode: str = None):
		self.trim_memory()
		if memory_sampling_mode in (None, "last"):
			batch_x = self.memory_x[-batch_size:]
			batch_y = self.memory_y[-batch_size:]
		elif memory_sampling_mode in ("uniform", "linear_skew"):
			batch_x = []
			batch_y = []
			if self.memory_sampling_dist is None or len(self.memory_x) != len(self.memory_sampling_dist):
				self.make_distribution(len(self.memory_x), memory_sampling_mode)
			indices = np.random.choice(
				np.arange(len(self.memory_x)),
				batch_size,
				p=self.memory_sampling_dist
			)
			for i in indices:
				batch_x.append(self.memory_x[i])
				batch_y.append(self.memory_y[i])
		else:
			raise ValueError(f"Invalid mode: '{memory_sampling_mode}'")
		return batch_x, batch_y

	def trim_memory(self, length=None):
		if not length:
			length = self.max_memory
		self.memory_x = self.memory_x[-length:]
		self.memory_y = self.memory_y[-length:]

	def make_distribution(self, size: int, mode: str = None):
		if not mode or mode == "uniform":
			self.memory_sampling_dist = np.ones(size) / size
		elif mode == "linear_skew":
			d = np.linspace(0, 1, size + 1)[1:]
			self.memory_sampling_dist = d / d.sum()
		# elif mode == "quadratic_skew":
		# 	d = np.linspace(0, 1, size + 1)[1:]
		# 	d = d * d
		# 	self.memory_sampling_dist = d / d.sum()
		else:
			raise ValueError(f"Invalid mode: '{mode}'")

	def load(self, name: str):
		self.model.load_weights(name)

	def save(self, name: str):
		self.model.save_weights(name)


class Sender(QAgent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model = build_sender_model(**kwargs)
		self.model_train = self.model
		self.reset_memory()


class Receiver(QAgent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model = build_receiver_model(**kwargs)
		self.model_train = self.model
		self.reset_memory()
