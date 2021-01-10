# from __future__ import annotations

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
LEARNING_RATE = 0.01
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


def build_sender_model(n_images: int,
                       input_image_shape: typing.Iterable,
                       embedding_size: int,
                       vocabulary_size: int,
                       optimizer: typing.Type[optim.Optimizer] = optimizer_dict["sgd"],
                       learning_rate: float = LEARNING_RATE,
                       sender_type: str = "agnostic",
                       verbose: bool = False,
                       image_embedding_layer: typing.Optional[layers.Layer] = None,
                       **kwargs) -> (models.Model, models.Model):
    if isinstance(optimizer, str):
        optimizer = optimizer_dict[optimizer.lower()]
    image_inputs = [layers.Input(shape=input_image_shape, name=f"image_in_{i}", dtype="float32")
                    for i in range(n_images)]
    if not image_embedding_layer:
        image_embedding_layer = layers.Dense(embedding_size, name="image_embedding")

    # agnostic part
    sigmoid = layers.Activation("sigmoid", name="sigmoid")
    output_layer = layers.Dense(vocabulary_size, name="output")

    # informed part
    stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="stack")
    permute = layers.Permute([2, 1], name="permute")
    feature_filters = layers.Conv1D(filters=vocabulary_size,
                                    kernel_size=(1,),
                                    input_shape=[n_images, embedding_size],
                                    activation="sigmoid",
                                    data_format="channels_last",
                                    name="feature_filters")
    vocabulary_filter = layers.Conv1D(1,
                                      kernel_size=(1,),
                                      data_format="channels_last",
                                      name="vocabulary_filter")
    flatten = layers.Flatten()

    temperature_input = layers.Input(shape=[], dtype="float32", name="temperature_input")

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

    model_predict = models.Model([*image_inputs, temperature_input], y, name="predict")

    index = layers.Input(shape=[1], dtype="int32", name="index_in")
    y_selected = layers.Lambda(
        lambda probs_index: tf.gather(*probs_index, axis=-1),
        name="gather")([y, index])

    def loss(target, prediction):
        return - K.log(prediction) * target

    model_train = models.Model([*image_inputs, index, temperature_input],
                               y_selected, name="train")
    model_train.compile(loss=loss, optimizer=optimizer(name="optim-sender", learning_rate=learning_rate))

    if verbose:
        model_predict.summary()
        model_train.summary()

    return model_predict, model_train


def build_receiver_model(n_images: int,
                         input_image_shape: typing.Iterable,
                         embedding_size: int,
                         vocabulary_size: int,
                         optimizer: typing.Type[optim.Optimizer] = optimizer_dict["sgd"],
                         learning_rate: float = LEARNING_RATE,
                         verbose: bool = False,
                         image_embedding_layer: typing.Optional[layers.Layer] = None,
                         **kwargs) -> (models.Model, models.Model):
    if isinstance(optimizer, str):
        optimizer = optimizer_dict[optimizer.lower()]
    image_inputs = [layers.Input(shape=input_image_shape, name=f"image_in_{i}", dtype="float32")
                    for i in range(n_images)]
    if not image_embedding_layer:
        image_embedding_layer = layers.Dense(embedding_size, name="image_embedding")

    temperature_input = layers.Input(shape=[], dtype="float32", name="temperature_input")
    softmax = layers.Softmax()

    symbol_input = layers.Input(shape=[1], dtype="int32", name=f"symbol_in")
    symbol_embedding = layers.Embedding(input_dim=vocabulary_size,
                                        output_dim=embedding_size,
                                        name="symbol_embedding")
    dot_product = layers.Dot(axes=-1, name="dot_product")

    y_images = [image_embedding_layer(x) for x in image_inputs]
    y_symbol = symbol_embedding(symbol_input)
    y = [dot_product([img, y_symbol]) for img in y_images]
    y = layers.concatenate(y, axis=-1)
    y = y / temperature_input
    y = softmax(y)

    model_predict = models.Model([*image_inputs, symbol_input, temperature_input], y, name="predict")

    index = layers.Input(shape=[1], dtype="int32", name="index_in")
    y_selected = layers.Lambda(
        lambda probs_index: tf.gather(*probs_index, axis=-1),
        name="gather")([y, index])

    def loss(target, prediction):
        return - K.log(prediction) * target

    model_train = models.Model([*image_inputs, symbol_input, index, temperature_input],
                               y_selected, name="train")
    model_train.compile(loss=loss, optimizer=optimizer(name="optim-receiver", learning_rate=learning_rate))

    if verbose:
        model_predict.summary()
        model_train.summary()

    return model_predict, model_train


class Agent:
    def __init__(self,
                 name: str,
                 role: str,
                 temperature: float = TEMPERATURE,
                 **kwargs: dict):
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

    def set_model(self,
                  model_predict: models.Model,
                  model_train: models.Model):
        self.model = model_predict
        self.model_train = model_train

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
        x.append(np.asarray([temperature]))
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
            x = [*x, np.array([temperature])]
            loss.append(self.model_train.train_on_batch(x=x, y=y))
        if reset_after:
            self.reset_memory()
        self.last_loss = np.mean(loss)
        return loss

    def get_last_loss(self, net_name: str = None):
        if not net_name or net_name == self.role:
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

    def _build_model(self, **kwargs):
        self.set_model(*build_sender_model(**kwargs))


class Receiver(Agent):
    def __init__(self,
                 name: str,
                 **kwargs):
        super(Receiver, self).__init__(name=name,
                                       role="receiver")
        self._build_model(**kwargs)
        self.reset_memory()

    def _build_model(self, **kwargs):
        self.set_model(*build_receiver_model(**kwargs))


class MultiAgent(Agent):
    """
    Includes both a sender network and a receiver network. Allows for switching between the two roles.
    """

    def __init__(self,
                 name: str,
                 role: str,
                 **kwargs):
        assert role in ("sender", "receiver")
        super(MultiAgent, self).__init__(name=name,
                                         role=role)
        self.name = name
        self.net: typing.Dict[str, typing.Optional[Agent]] = {
            "sender": None,
            "receiver": None
        }
        self._build_model(**kwargs)
        self.reset_memory()

    def _build_model(self, shared_embedding, embedding_size, **kwargs):
        print(f"shared_embedding: {shared_embedding}")
        for k, v in kwargs.items():
            print(f"{k}: {v}")
        if shared_embedding:
            image_embedding_layer = layers.Dense(embedding_size, name="shared_image_embedding")
        else:
            image_embedding_layer = None

        sender_update = {}
        receiver_update = {}
        if "sender_settings" in kwargs:
            sender_update = kwargs.pop("sender_settings") or {}
        if "receiver_settings" in kwargs:
            receiver_update = kwargs.pop("receiver_settings") or {}
        sender_update["image_embedding_layer"] = image_embedding_layer
        receiver_update["image_embedding_layer"] = image_embedding_layer
        sender_update["embedding_size"] = embedding_size
        receiver_update["embedding_size"] = embedding_size
        sender_kwargs = {**kwargs, **sender_update}
        receiver_kwargs = {**kwargs, **receiver_update}

        self.net["sender"] = Agent(name="sender_rf", role="sender")
        self.net["receiver"] = Agent(name="receiver_rf", role="receiver")
        self.net["sender"].set_model(*build_sender_model(**sender_kwargs))
        self.net["receiver"].set_model(*build_receiver_model(**receiver_kwargs))

    def switch_role(self):
        if self.role == "sender":
            self.role = "receiver"
        else:
            self.role = "sender"

    def active_net(self):
        return self.net[self.role]

    def act(self, state: list, **kwargs):
        return self.active_net().act(state, **kwargs)

    def batch_train(self):
        self.active_net().batch_train()

    def reset_batch(self):
        self.active_net().reset_batch()

    def prepare_batch(self, *args, **kwargs):
        pass

    def load(self, name: str):
        self.net["sender"].model.load_weights(f"{name}.snd")
        self.net["receiver"].model.load_weights(f"{name}.rcv")

    def save(self, name: str):
        self.net["sender"].model.save_weights(f"{name}.snd")
        self.net["receiver"].model.save_weights(f"{name}.rcv")

    def predict(self, state: list, temperature: typing.Optional[float] = None):
        return self.active_net().predict(state, temperature)

    def update(self, state: list, action: np.array, target: np.array, temperature: typing.Optional[float] = None):
        return self.active_net().update(state, action, target, temperature)

    def remember(self, state: list, action: np.array, target: np.array):
        self.active_net().remember(state, action, target)

    def reset_memory(self):
        self.active_net().reset_memory()

    def update_on_batch(self, reset_after: bool = True, temperature: typing.Optional[float] = None):
        return self.active_net().update_on_batch(reset_after, temperature)

    def get_last_loss(self, net_name: str = None):
        if not net_name:
            return self.active_net().get_last_loss()
        elif net_name in self.net:
            return self.net[net_name].last_loss
        else:
            return np.NaN

    def set_temperature(self, temperature: float):
        self.active_net().set_temperature(temperature)

    def adjust_temperature(self, loss: float):
        self.active_net().adjust_temperature(loss)
