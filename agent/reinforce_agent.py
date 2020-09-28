import typing

import numpy as np
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optim
import tensorflow.keras.backend as K

import sys
sys.path.append("..")

from utils.debug import get_shape
from utils.utils import dict_default

N_IMAGES = 2
EMBEDDING_SIZE = 50
VOCABULARY_SIZE = 100
TEMPERATURE = 10
SENDER_TYPE = "agnostic"
LEARNING_RATE = 0.1
OPTIMIZER = "adam"
SHARED_EMBEDDING = False
ADAPTIVE_TEMPERATURE = False

optimizer_dict = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}


def convert_parameter_dict(old_params: dict):
    new_params = {
        "n_images": len(old_params["input_shapes"]) - 1,
        "input_image_shape": old_params["input_shapes"][0],
        "embedding_size": dict_default(old_params, "embedding_size", EMBEDDING_SIZE),
        "vocabulary_size": dict_default(old_params, "n_symbols", VOCABULARY_SIZE),
        "sender_type": dict_default(old_params, "sender_type", SENDER_TYPE),
        "temperature": dict_default(old_params, "gibbs_temperature", TEMPERATURE),
        "shared_embedding": dict_default(old_params, "shared_embedding", SHARED_EMBEDDING),
        "learning_rate": dict_default(old_params, "learning_rate", LEARNING_RATE),
        "adaptive_temperature": dict_default(old_params, "adaptive_temperature", ADAPTIVE_TEMPERATURE),
        "optimizer": dict_default(old_params,
                                  "optimizer",
                                  optimizer_dict[OPTIMIZER],
                                  lambda x: optimizer_dict[x.lower()])
    }
    return new_params


class Agent:
    def __init__(self,
                 name: str,
                 role: str,
                 temperature: float = TEMPERATURE,
                 **kwargs):
        self.name: str = name
        self.role: str = role
        self.model: typing.Optional[models.Model] = None
        self.model_train: typing.Optional[models.Model] = None
        self.memory_x: list = []
        self.memory_y: list = []
        self.temperature: float = temperature
        self.last_loss: typing.Optional[float] = None

    def _build_model(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, state: list, **kwargs):
        probs = np.squeeze(self.predict(state))
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action, probs

    def batch_train(self):
        self.update_on_batch(reset_after=True)

    def reset_batch(self):
        self.reset_memory()

    def prepare_batch(self, *args, **kwargs):
        pass

    def load(self, name: str):
        self.model.load_weights(name)

    def save(self, name: str):
        self.model.save_weights(name)

    def predict(self, state: list, temperature: typing.Optional[float] = None):
        if temperature:
            assert temperature > 0
        else:
            temperature = self.temperature
        x = [np.expand_dims(s, axis=0) for s in state]
        x.append(np.asarray([temperature]))
        # print(get_shape(x))
        return self.model.predict_on_batch(x)

    def update(self, state: list, action: np.array, target: np.array, temperature: typing.Optional[float] = None):
        if temperature:
            assert temperature > 0
        else:
            temperature = self.temperature
        x = [np.expand_dims(s, axis=0) for s in state]
        x.append(np.asarray([action]))
        x.append(np.asarray([self.temperature]))
        # print(get_shape(x), get_shape(target))
        loss = self.model_train.train_on_batch(x=x, y=target)
        self.last_loss = loss
        return loss

    def remember(self, state: list, action: np.array, target: np.array):
        x = [np.expand_dims(s, axis=0) for s in state]
        x.append(np.asarray([action]))
        self.memory_x.append(x)
        self.memory_y.append(np.asarray([target]))

    def reset_memory(self):
        self.memory_x = []
        self.memory_y = []

    def update_on_batch(self, reset_after: bool = True, temperature: typing.Optional[float] = None):
        if temperature:
            assert temperature > 0
        else:
            temperature = self.temperature
        loss = []
        for x, y in zip(self.memory_x, self.memory_y):
            x = [*x, np.array([self.temperature])]
            loss.append(self.model_train.train_on_batch(x=x, y=y))
        if reset_after:
            self.reset_memory()
        self.last_loss = np.mean(loss)
        return loss

    def get_last_loss(self, net_name: str = None):
        if net_name == self.role:
            return self.last_loss
        else:
            return np.NaN

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def adjust_temperature(self, loss: float):
        if loss < 0.001:
            self.temperature *= 1.1
        elif loss > 2:
            self.temperature /= 1.1


class Sender(Agent):
    def __init__(self,
                 name: str,
                 **kwargs):
        super(Sender, self).__init__(name=name,
                                     role="sender")
        self._build_model(**kwargs)
        self.reset_memory()

    def _build_model(self, n_images, input_image_shape, embedding_size, vocabulary_size, optimizer,
                    learning_rate, sender_type="agnostic", verbose=False, **kwargs):
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
            y = permute(y)
            y = vocabulary_filter(y)
            y = flatten(y)

        y = y / temperature_input
        y = softmax(y)

        self.model = models.Model([*image_inputs, temperature_input], y, name="S_predict")

        index = layers.Input(shape=[1], dtype="int32", name="S_index_in")
        y_selected = layers.Lambda(
            lambda probs_index: K.gather(probs_index[0][0], probs_index[1]),
            name="S_gather")([y, index])

        def loss(target, prediction):
            return - K.log(prediction) * target

        self.model_train = models.Model([*image_inputs, index, temperature_input],
                                 y_selected, name="S_train")
        self.model_train.compile(loss=loss, optimizer=optimizer(learning_rate))

        if verbose:
            self.model.summary()
            self.model_train.summary()


class Receiver(Agent):
    def __init__(self,
                 name: str,
                 **kwargs):
        super(Receiver, self).__init__(name=name,
                                       role="receiver")
        self._build_model(**kwargs)
        self.reset_memory()

    def _build_model(self, n_images, input_image_shape, embedding_size, vocabulary_size, optimizer,
                     learning_rate, verbose=False, **kwargs):
        image_inputs = [layers.Input(shape=input_image_shape, name=f"R_image_in_{i}", dtype="float32")
                        for i in range(n_images)]
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

        self.model = models.Model([*image_inputs, symbol_input, temperature_input], y, name="R_predict")

        index = layers.Input(shape=[1], dtype="int32", name="R_index_in")
        y_selected = layers.Lambda(
            lambda probs_index: tf.gather(*probs_index, axis=-1),
            name="R_gather")([y, index])

        # @tf.function
        def loss(target, prediction):
            return - K.log(prediction) * target

        self.model_train = models.Model([*image_inputs, symbol_input, index, temperature_input],
                                 y_selected, name="R_train")
        self.model_train.compile(loss=loss, optimizer=optimizer(learning_rate))

        if verbose:
            self.model.summary()
            self.model_train.summary()
