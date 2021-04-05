import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import backend as K


def getshape(array):
	if isinstance(array, list):
		return [getshape(x) for x in array]
	else:
		try:
			return array.shape
		except:
			return None


def build_sender_model(n_images, input_image_shape, embedding_size,
		vocabulary_size, optimizer,
		sender_type="agnostic",
		verbose=False):
	image_inputs = [layers.Input(shape=input_image_shape,
								 name=f"S_image_in_{i}",
								 dtype="float32") for i in range(n_images)]
	image_embedding_layer = layers.Dense(embedding_size,
										 name="S_image_embedding")
	# agnostic part
	sigmoid = layers.Activation("sigmoid", name="S_sigmoid")
	output_layer = layers.Dense(vocabulary_size, name="S_output")

	# informed part
	stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="S_stack")
	permute = layers.Permute([2, 1], name="S_permute")
	feature_filters = layers.Conv1D(filters=vocabulary_size,
									kernel_size=(1,),
									input_shape=[n_images, embedding_size],
									activation="sigmoid",
									data_format="channels_last",
									name="S_feature_filters")

	permute_2 = layers.Permute([2, 1], name="S_permute_2")
	vocabulary_filter = layers.Conv1D(1,
									  kernel_size=(1,),
									  data_format="channels_last",
									  name="S_vocabulary_filter")
	flatten = layers.Flatten()

	temperature_input = layers.Input(shape=[], dtype="float32", name="R_temperature_input")

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

	y = y / temperature_input
	y = softmax(y)

	model_predict = models.Model([*image_inputs, temperature_input], y, name="S_predict")

	index = layers.Input(shape=[1], dtype="int32", name="S_index_in")
	y_selected = layers.Lambda(
		lambda probs_index: K.gather(probs_index[0][0], probs_index[1]),
		name="S_gather")([y, index])

	def loss(target, prediction):
		return - K.log(prediction) * target

	model_train = models.Model([*image_inputs, index, temperature_input],
									y_selected, name="S_train")
	model_train.compile(loss=loss, optimizer=optimizer)

	if verbose:
		model_predict.summary()
		model_train.summary()

	return model_predict, model_train


def build_receiver_model(n_images, input_image_shape, embedding_size,
				vocabulary_size, optimizer, verbose=False):
	image_inputs = [layers.Input(shape=input_image_shape, name=f"R_image_in_{i}", dtype="float32") for i in
					range(n_images)]
	image_embedding_layer = layers.Dense(embedding_size, name="R_image_embedding")

	temperature_input = layers.Input(shape=[], dtype="float32", name="R_temperature_input")
	softmax = layers.Softmax()

	symbol_input = layers.Input(shape=[1], dtype="int32", name=f"R_symbol_in")
	symbol_embedding = layers.Embedding(input_dim=vocabulary_size,
										output_dim=embedding_size,
										name="R_symbol_embedding")
	dot_product = layers.Dot(axes=-1, name="R_dot_product")

	y_images = [image_embedding_layer(x) for x in image_inputs]
	y_symbol = symbol_embedding(symbol_input)
	y = [dot_product([img, y_symbol]) for img in y_images]
	y = layers.concatenate(y, axis=-1)
	y = y / temperature_input
	y = softmax(y)

	model_predict = models.Model([*image_inputs, symbol_input, temperature_input], y, name="R_predict")

	index = layers.Input(shape=[1], dtype="int32", name="R_index_in")
	y_selected = layers.Lambda(
		lambda probs_index: tf.gather(*probs_index, axis=-1),
		name="R_gather")([y, index])

	# @tf.function
	def loss(target, prediction):
		return - K.log(prediction) * target

	model_train = models.Model([*image_inputs, symbol_input, index, temperature_input],
							 y_selected, name="R_train")
	model_train.compile(loss=loss, optimizer=optimizer)

	if verbose:
		model_predict.summary()
		model_train.summary()

	return model_predict, model_train


class Agent:
	def __init__(self, temperature=1, **kwargs):
		self.temperature = temperature
		self.model, self.model_train = None, None

	def predict(self, state):
		x = [*state, np.array([self.temperature])]
		return self.model.predict_on_batch(x)

	def update(self, state, action, target):
		x = [*state, action, np.array(self.temperature)]
		print(getshape(x), getshape(target))
		return self.model_train.train_on_batch(x=x, y=target)

	def remember(self, state, action, target):
		x = [*state, action]
		self.memory_x.append(x)
		self.memory_y.append(target)

	def reset_memory(self):
		self.memory_x = []
		self.memory_y = []

	def update_on_batch(self, reset_after=True):
		loss = []
		for x, y in zip(self.memory_x, self.memory_y):
			x = [*x, np.array([self.temperature])]
			loss.append(self.model_train.train_on_batch(x=x, y=y))
		if reset_after:
			self.reset_memory()
		return loss

	def set_temperature(self, temperature):
		self.temperature = temperature

	def load(self, name: str):
		self.model.load_weights(name)

	def save(self, name: str):
		self.model.save_weights(name)


class Sender(Agent):
	def __init__(self, temperature=1, **kwargs):
		super().__init__(temperature, **kwargs)
		self.model, self.model_train = build_sender_model(**kwargs)
		self.reset_memory()


class Receiver(Agent):
	def __init__(self, temperature=1, **kwargs):
		super().__init__(temperature, **kwargs)
		self.model, self.model_train = build_receiver_model(**kwargs)
		self.reset_memory()
