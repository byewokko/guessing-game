import numpy as np
import scipy.stats as stats
from collections import deque
from keras.models import Sequential, Model
import keras.layers as layers
from keras.applications import MobileNetV2
import keras.backend as backend
from keras.losses import categorical_crossentropy


def sample_memory(memory, batch_size, mode="uniform"):
    if len(memory) < batch_size:
        return

    data = []

    if mode == "tail":
        idx = np.arange(len(memory) - batch_size, len(memory))  # take just the newest exps
    elif mode == "norm-tail":
        sd = 0.4
        dist = stats.norm.pdf(np.linspace(-1, 0, len(memory)), 0, sd)  # sample from the left half of normdist
        dist /= dist.sum()
        idx = np.random.choice(np.arange(len(memory)), size=batch_size, replace=False, p=dist)
    else:  # if mode == "uniform"
        idx = np.random.choice(np.arange(len(memory)), size=batch_size, replace=False)

    for i in idx:
        data.append(memory[i])

    return data


class Sender:
    """
    Uses Q-values
    """
    input_type = "data"

    def __init__(self, n_symbols, name="", **kwargs):
        self.n_symbols = n_symbols
        self.n_input_images = 2
        self.memory = deque(maxlen=255)
        self.last_action = None
        self.model = None
        self.name = name
        self.expl_rate = False
        self.expl_rate_decay = 0.95
        self.build_model(n_symbols=n_symbols, **kwargs)

    def build_model(self, image_shape, n_symbols):
        img_input_0 = layers.Input(shape=image_shape, name='image_0')
        img_input_1 = layers.Input(shape=image_shape, name='image_1')
        vision = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_shape)
        flatten = layers.Flatten()
        cnn_out = layers.Dense(128, activation='relu')

        img_0 = cnn_out(flatten(vision(img_input_0)))
        img_1 = cnn_out(flatten(vision(img_input_1)))

        concat = layers.concatenate([img_0, img_1], axis=-1)
        out = layers.Dense(n_symbols, activation='linear')(concat)
        self.model = Model([img_input_0, img_input_1], out)

        self.model.compile(optimizer='adam', loss="mse")

    def act_send(self, images):
        assert len(images) == 2
        data = [np.expand_dims(images[0], 0), np.expand_dims(images[1], 0)]
        if self.expl_rate and np.random.random() < self.expl_rate:
            self.expl_rate *= self.expl_rate_decay
            clue = np.random.randint(0, self.n_symbols)
        else:
            estimated = self.model.predict(data)
            clue = np.argmax(estimated)
        self.last_action = (data, clue)
        return clue

    def reward_send(self, reward):
        self.memory.append((*self.last_action, reward))

    def experience_replay(self, n=10, sampling_mode="norm-tail"):
        batch = sample_memory(self.memory, n, sampling_mode)
        if not batch:
            print("Not enough experience. Skipping training.")
            return
        else:
            for state, action, reward in batch:
                q_values = self.model.predict(state)
                q_values[0][action] = reward
                self.model.fit(state, q_values, verbose=0)


class Receiver:
    input_type = "data"

    def __init__(self, n_symbols, name="", **kwargs):
        self.n_symbols = n_symbols
        self.n_input_images = 2
        self.memory = deque(maxlen=255)
        self.last_action = None
        self.model = None
        self.name = name
        self.expl_rate = False
        self.expl_rate_decay = 0.95
        self.build_model(n_symbols=n_symbols, n_images=self.n_input_images, **kwargs)

    def build_model(self, image_shape, n_symbols, n_images):
        img_input_0 = layers.Input(shape=image_shape, name='image_0')
        img_input_1 = layers.Input(shape=image_shape, name='image_1')
        vision = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_shape)
        flatten = layers.Flatten()
        cnn_out = layers.Dense(128, activation='relu')

        img_0 = cnn_out(flatten(vision(img_input_0)))
        img_1 = cnn_out(flatten(vision(img_input_1)))

        sym_input = layers.Input(shape=(1,), dtype="int32", name="symbol")
        embed = layers.Embedding(input_dim=n_symbols, output_dim=32)
        resh = layers.Reshape((32,))
        symbol = resh(embed(sym_input))

        concat = layers.concatenate([img_0, img_1, symbol], axis=-1)
        out = layers.Dense(n_symbols, activation='linear')(concat)
        self.model = Model([img_input_0, img_input_1, sym_input], out)

        self.model.compile(optimizer='adam', loss="mse")

    def act_receive(self, images, clue):
        assert len(images) == 2
        data = [np.expand_dims(images[0], 0), np.expand_dims(images[1], 0), np.expand_dims(clue, 0)]
        if self.expl_rate and np.random.random() < self.expl_rate:
            self.expl_rate *= self.expl_rate_decay
            pick = np.random.randint(0, self.n_input_images)
        else:
            estimated = self.model.predict(data)
            pick = np.argmax(estimated)
        self.last_action = (data, pick)
        return clue

    def reward_guess(self, reward):
        self.memory.append((*self.last_action, reward))

    def experience_replay(self, n=10, sampling_mode="norm-tail"):
        batch = sample_memory(self.memory, n, sampling_mode)
        if not batch:
            print("Not enough experience. Skipping training.")
            return
        else:
            for state, action, reward in batch:
                q_values = self.model.predict(state)
                q_values[0][action] = reward
                self.model.fit(state, q_values, verbose=0)