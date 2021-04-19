import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import backend as K

from .agent import Agent

L = logging.getLogger(__name__)


def build_sender_model(
		n_images, input_image_shape, embedding_size,
		vocabulary_size, optimizer, temperature,
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

	softmax = layers.Softmax()

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

	y = y / temperature
	y = softmax(y)

	model_predict = models.Model(image_inputs, y, name="S_predict")

	index = layers.Input(shape=[1], dtype="int32", name="S_index_in")
	y_selected = layers.Lambda(
		lambda probs_index: K.gather(probs_index[0][0], probs_index[1]),
		name="S_gather"
	)([y, index])

	@tf.function
	def loss(target, prediction):
		return - K.log(prediction) * target

	model_train = models.Model(
		[*image_inputs, index],
		y_selected, name="S_train"
	)
	model_train.compile(loss=loss, optimizer=optimizer)

	if verbose:
		model_predict.summary()
		model_train.summary()

	return model_predict, model_train


def build_receiver_model(
		n_images, input_image_shape, embedding_size, temperature,
		vocabulary_size, optimizer, image_embedding_layer=None, verbose=False, **kwargs
):
	image_inputs = [
		layers.Input(shape=input_image_shape, name=f"R_image_in_{i}", dtype="float32")
		for i
		in range(n_images)
	]

	if not image_embedding_layer:
		image_embedding_layer = layers.Dense(embedding_size, name="R_image_embedding")

	softmax = layers.Softmax()

	symbol_input = layers.Input(shape=[1], dtype="int32", name=f"R_symbol_in")
	symbol_embedding = layers.Embedding(
		input_dim=vocabulary_size,
		output_dim=embedding_size,
		name="R_symbol_embedding"
	)
	dot_product = layers.Dot(axes=-1, name="R_dot_product")

	y_images = [image_embedding_layer(x) for x in image_inputs]
	y_symbol = symbol_embedding(symbol_input)
	y = [dot_product([img, y_symbol]) for img in y_images]
	y = layers.concatenate(y, axis=-1)
	y = y / temperature
	y = softmax(y)

	model_predict = models.Model([*image_inputs, symbol_input], y, name="R_predict")

	index = layers.Input(shape=[1], dtype="int32", name="R_index_in")
	y_selected = layers.Lambda(
		lambda probs_index: tf.gather(*probs_index, axis=-1),
		name="R_gather"
	)([y, index])

	@tf.function
	def loss(target, prediction):
		return - K.log(prediction) * target

	model_train = models.Model(
		[*image_inputs, symbol_input, index],
		y_selected,
		name="R_train"
	)
	model_train.compile(loss=loss, optimizer=optimizer)

	if verbose:
		model_predict.summary()
		model_train.summary()

	return model_predict, model_train


class Sender(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model, self.model_train = build_sender_model(**kwargs)
		self.reset_memory()


class Receiver(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model, self.model_train = build_receiver_model(**kwargs)
		self.reset_memory()
