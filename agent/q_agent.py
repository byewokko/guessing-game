import numpy as np
from datetime import datetime as dt
from keras.models import Model
import keras.layers as layers
import keras.optimizers as optim
import keras.backend as K
import keras.initializers as init
from keras.losses import categorical_crossentropy
from keras.regularizers import l2

from agent import component
from utils.debug import print_layer

GIBBS_TEMPERATURE = 0.05
USE_BIAS = True
LEARNING_RATE = 0.005
EMBEDDING_SIZE = 50
OPTIMIZER = optim.Adam
LOSS = "mse"
CNN_FILTERS = 20
EXPLORATION_RATE = 1.
EXPLORATION_DECAY = .99
EXPLORATION_FLOOR = .015


class Agent:
    input_type = "data"

    def __init__(self, input_shapes, output_size, n_symbols, gibbs_temperature=GIBBS_TEMPERATURE, **kwargs):
        self.last_updates = None
        self.n_input_images = 2
        self.input_shapes = input_shapes
        self.output_size = output_size
        self.n_symbols = n_symbols
        self.last_action = None
        self.last_loss = 0
        self.last_weights = None
        self.role = None
        self.model = None
        self.model_predict = None
        self.gibbs_temperature = gibbs_temperature
        self.exploration_rate = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_DECAY
        self.exploration_min = EXPLORATION_FLOOR
        self.batch_states = None
        self.batch_actions = None
        self.batch_rewards = None
        self.reset_batch()

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state, explore="gibbs", gibbs_temperature=None):
        assert explore in (False, "gibbs", "decay")
        if gibbs_temperature is None:
            gibbs_temperature = self.gibbs_temperature
        else:
            assert gibbs_temperature > 0
        state = [np.expand_dims(st, 0) for st in state]
        # action = np.zeros(self.output_size)
        act_probs = self.model.predict(state)
        if np.any(np.isnan(act_probs)):
            raise ValueError("probabilities contain NaN")
        act_probs = np.squeeze(act_probs)
        if explore == "gibbs":
            # Sample from Gibbs distribution
            act_probs_exp = np.exp(act_probs / gibbs_temperature)
            act_probs = act_probs_exp / act_probs_exp.sum()
            action = np.random.choice(range(len(act_probs)), 1, p=act_probs)
        elif explore == "decay":
            if np.random.rand() > self.exploration_rate:
                if self.exploration_rate > self.exploration_min:
                    self.exploration_rate *= self.exploration_rate_decay
                action = np.random.choice(range(len(act_probs)), 1)
            else:
                action = np.argmax(act_probs)
        else:
            action = np.argmax(act_probs)
        self.last_action = (state, action, act_probs)
        return action, act_probs

    def fit(self, state, action, reward):
        # self.last_weights = self.model.get_weights()
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(action_onehot, 0)
        self.last_loss = self.model.train_on_batch([*X, reward], Y)
        # self.last_updates = self.optimizer.get_updates() # needs args: loss and params

    def remember(self, state, action, reward):
        for i in range(len(state)):
            self.batch_states[i].append(state[i])
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        self.batch_actions.append(action_onehot)
        self.batch_rewards.append(reward)

    def batch_train(self):
        q_values = self.model.predict(self.batch_states)
        for i in range(len(self.batch_rewards)):
            q_values[i][self.batch_actions[i].astype("bool")] = self.batch_rewards[i]
        self.last_loss = self.model.train_on_batch(
            self.batch_states,
            q_values
        )
        self.reset_batch()

    def reset_batch(self):
        self.batch_states = [[] for _ in self.input_shapes]
        self.batch_actions = []
        self.batch_rewards = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Sender(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.role = "sender"
        self._build_model(**kwargs)

    def _build_model(self, input_shapes, output_size, embedding_size=EMBEDDING_SIZE,
                     use_bias=USE_BIAS, loss=LOSS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     **kwargs):
        n_inputs = len(input_shapes)
        inputs = [layers.Input(shape=input_shapes[i],
                               name=f"input_{i}")
                  for i in range(n_inputs)]
        embs = [layers.Dense(embedding_size,
                             activation="linear",
                             use_bias=use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        # imgs = [embs[i](inputs[i]) for i in range(n_inputs)]  # separate embedding layer for each image
        imgs = [embs[0](inputs[i]) for i in range(n_inputs)]  # same embedding layer for all images

        concat = layers.concatenate(imgs, axis=-1)
        # hidden = layers.Dense(self.embedding_size,
        #                    activation='relu',
        #                    use_bias=self.use_bias,
        #                    # kernel_initializer=init.glorot_uniform(seed=42)
        #                    )(concat)
        out = layers.Dense(output_size,
                           activation='sigmoid',
                           use_bias=use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42)
                           )(concat)

        self.model = Model(inputs, out)
        self.model.compile(loss=loss, optimizer=optimizer(lr=learning_rate))


class SenderInformed(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.role = "sender"
        self._build_model(**kwargs)

    def _build_model(self, input_shapes, output_size, n_filters=CNN_FILTERS, embedding_size=EMBEDDING_SIZE,
                     use_bias=USE_BIAS, loss=LOSS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     **kwargs):
        n_inputs = len(self.input_shapes)
        inputs = [layers.Input(shape=self.input_shapes[i],
                               name=f"input_{i}")
                  for i in range(n_inputs)]
        embs = [layers.Dense(embedding_size,
                             activation='linear',
                             use_bias=use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        emb = layers.Dense(embedding_size,
                           activation='linear',
                           use_bias=use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42),
                           # kernel_regularizer=l2(0.001),
                           name=f"embed_img")

        imgs = [embs[i](inputs[i]) for i in range(n_inputs)]  # separate embedding layer for each image
        # imgs = [emb(inputs[i]) for i in range(n_inputs)]  # same embedding layer for all images

        stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="stack")
        reshape = layers.Reshape((-1, embedding_size, 1))
        feat_filters = layers.Conv2D(filters=n_filters,
                                     kernel_size=(n_inputs, 1),
                                     activation="sigmoid",
                                     # activation="elu",
                                     # padding="same",
                                     # strides=embedding_size,
                                     # data_format="channels_last",
                                     name="feature_filters"
                                     )

        voc_filter = layers.Conv2D(1, (1, n_filters),
                                   # padding="same",
                                   activation="linear",
                                   data_format="channels_first",
                                   name="vocab_filter"
                                   )

        dense = layers.Dense(output_size, activation="softmax", name="output_dense")

        out = dense(layers.Flatten()(voc_filter(feat_filters(reshape(stack(imgs))))))

        self.model = Model(inputs, out)
        self.model.compile(loss=loss, optimizer=optimizer(lr=learning_rate))


class Receiver(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.role = "receiver"
        self._build_model(**kwargs)

    def _build_model(self, input_shapes, output_size, embedding_size=EMBEDDING_SIZE,
                     use_bias=USE_BIAS, loss=LOSS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     mode="dot", **kwargs):
        n_input_images = len(input_shapes) - 1
        inputs = [layers.Input(shape=input_shapes[i],
                               name=f"input_{i}")
                  for i in range(n_input_images)]
        embs = [layers.Dense(embedding_size,
                             activation='linear',
                             use_bias=use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_input_images)]
        emb = layers.Dense(embedding_size,
                           activation='linear',
                           use_bias=use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42),
                           # kernel_regularizer=l2(0.001),
                           name=f"embed_img")

        # imgs = [embs[i](inputs[i]) for i in range(n_input_images)]  # separate embedding layer for each image
        imgs = [emb(inputs[i]) for i in range(n_input_images)]  # same embedding layer for all images

        symbol_shape = input_shapes[-1]
        sym_input = layers.Input(shape=symbol_shape, dtype="int32", name="input_sym")
        emb_sym = layers.Embedding(input_dim=self.n_symbols,
                                   output_dim=embedding_size,
                                   name=f"embed_sym")
        flat = layers.Flatten()
        symbol = flat(emb_sym(sym_input))

        if mode == "dot":
            dot = layers.Dot(axes=1)
            dot_prods = [dot([img, symbol]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
        elif mode == "dense":
            out = layers.concatenate([*imgs, symbol], axis=-1)
            out = layers.Dense(embedding_size,
                               # activation=layers.PReLU(),
                               # kernel_initializer=init.glorot_uniform(seed=42),
                               name=f"dense_join")(out)
            out = layers.Dense(output_size,
                               # kernel_initializer=init.glorot_uniform(seed=42),
                               name=f"dense_out")(out)
        else:
            raise ValueError(f"'{mode}' is not a valid mode.")

        out = layers.Activation("sigmoid")(out)

        self.model = Model([*inputs, sym_input], out)
        self.model.compile(loss=loss, optimizer=optimizer(lr=learning_rate))


class MultiAgent(Agent):
    def __init__(self, role="sender", name="agent", **kwargs):
        super().__init__(**kwargs)
        self.role = None
        self.name = name
        self.net = {"sender": component.Net(),
                    "receiver": component.Net()}
        self._build_model(**kwargs)
        self.set_role(role)

    def active_net(self):
        assert self.role in self.net.keys()
        return self.net[self.role]

    def _build_model(self, input_shapes, n_symbols, embedding_size=EMBEDDING_SIZE,
                     use_bias=USE_BIAS, loss=LOSS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     mode="dot", sender_type="agnostic", **kwargs):
        # Shared part
        n_inputs = len(input_shapes)
        n_input_images = len(input_shapes) - 1
        inputs = [layers.Input(shape=input_shapes[i],
                               name=f"input_{i}")
                  for i in range(n_input_images)]
        embs = [layers.Dense(embedding_size,
                             activation='linear',
                             use_bias=use_bias,
                             name=f"embed_{i}")
                for i in range(n_input_images)]
        emb = layers.Dense(embedding_size,
                           activation='linear',
                           use_bias=use_bias,
                           name=f"embed_img")

        imgs = [embs[i](inputs[i]) for i in range(n_input_images)]  # separate embedding layer for each image
        # imgs = [emb(inputs[i]) for i in range(n_input_images)]  # same embedding layer for all images

        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                optimizer = optim.Adam
            else:
                raise TypeError(f"Unknown optimizer '{optimizer}'")

        # Sender part
        concat = layers.concatenate(imgs, axis=-1)
        out = layers.Dense(n_symbols,
                           activation='sigmoid',
                           use_bias=use_bias,
                           )(concat)

        self.net["sender"].model = Model(inputs, out)
        self.net["sender"].model.compile(loss=loss, optimizer=optimizer(lr=learning_rate, clipnorm=1.))
        self.net["sender"].input_shapes = input_shapes[:-1]
        self.net["sender"].output_size = n_symbols
        self.net["sender"].reset_batch()

        # Receiver part
        symbol_shape = input_shapes[-1]
        sym_input = layers.Input(shape=symbol_shape, dtype="int32", name="input_sym")
        emb_sym = layers.Embedding(input_dim=n_symbols,
                                   output_dim=embedding_size,
                                   name=f"embed_sym")
        flat = layers.Flatten()
        symbol = flat(emb_sym(sym_input))

        if mode == "dot":
            dot = layers.Dot(axes=1)
            dot_prods = [dot([img, symbol]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
        elif mode == "dense":
            out = layers.concatenate([*imgs, symbol], axis=-1)
            out = layers.Dense(embedding_size,
                               # activation=layers.PReLU(),
                               # kernel_initializer=init.glorot_uniform(seed=42),
                               name=f"dense_join")(out)
            out = layers.Dense(len(imgs),
                               # kernel_initializer=init.glorot_uniform(seed=42),
                               name=f"dense_out")(out)
        else:
            raise ValueError(f"'{mode}' is not a valid mode.")

        out = layers.Activation("sigmoid")(out)

        self.net["receiver"].model = Model([*inputs, sym_input], out)
        self.net["receiver"].model.compile(loss=loss, optimizer=optimizer(lr=learning_rate, clipnorm=1.))
        self.net["receiver"].input_shapes = input_shapes
        self.net["receiver"].output_size = n_input_images
        self.net["receiver"].reset_batch()

    def set_role(self, role="sender"):
        assert role in ("sender", "receiver")
        self.role = role

    def switch_role(self):
        if self.role == "sender":
            self.set_role("receiver")
        else:
            self.set_role("sender")

    def act(self, state, explore="gibbs", gibbs_temperature=0.05):
        assert explore in (False, "gibbs", "decay")
        state = [np.expand_dims(st, 0) for st in state]
        # action = np.zeros(self.output_size)
        act_probs = self.active_net().predict(state)
        act_probs = np.squeeze(act_probs)
        if explore == "gibbs":
            # Sample from Gibbs distribution
            act_probs_exp = np.exp(act_probs / gibbs_temperature)
            # Random idea:
            # max_prob = act_probs.max()
            # act_probs_exp = np.exp(act_probs / max_prob)
            act_probs_exp = act_probs_exp / act_probs_exp.sum()
            # print(self.name, self.active_net().model.name, act_probs_exp.min(), act_probs_exp.max())
            action = np.random.choice(range(len(act_probs_exp)), 1, p=act_probs_exp)
        elif explore == "decay":
            if np.random.rand() > self.exploration_rate:
                if self.exploration_rate > self.exploration_min:
                    self.exploration_rate *= self.exploration_rate_decay
                action = np.random.choice(range(len(act_probs)), 1)
            else:
                action = np.argmax(act_probs)
        else:
            action = np.argmax(act_probs)
        self.last_action = (state, action, act_probs)
        return action, act_probs

    def fit(self, state, action, reward):
        # self.last_weights = self.model.get_weights()
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(action_onehot, 0)
        self.last_loss = self.active_net().train_on_batch([*X, reward], Y)
        # self.last_updates = self.optimizer.get_updates() # needs args: loss and params

    def remember(self, state, action, reward):
        self.active_net().remember(state, action, reward)

    def batch_train(self):
        self.last_loss = self.active_net().batch_train()

    def load(self, name):
        self.net["sender"].model.load_weights(f"{name}.snd")
        self.net["receiver"].model.load_weights(f"{name}.rcv")

    def save(self, name=None):
        if not name:
            time = dt.now().strftime("%y%m%d-%H%M")
            name = f"model-{time}"
        self.net["sender"].model.save_weights(f"{name}.snd")
        self.net["receiver"].model.save_weights(f"{name}.rcv")
