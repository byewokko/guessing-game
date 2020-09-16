import numpy as np
from datetime import datetime as dt
from keras.models import Model
import keras.layers as layers
import keras.optimizers as optim
import keras.backend as K
# import tensorflow.nn as N

from agent import component

GIBBS_TEMPERATURE = 1
USE_BIAS = True
LEARNING_RATE = 0.005
EMBEDDING_SIZE = 50
OPTIMIZER = optim.Adam
LOSS = "mse"
CNN_FILTERS = 20
EXPLORATION_RATE = 1.
EXPLORATION_DECAY = .99995
EXPLORATION_FLOOR = .02
MAX_MEMORY = 2000


def softmax(X, theta=1.0):
    y = X * theta
    y = y - np.max(y)
    y = np.exp(y)
    y = y / np.sum(y)
    return y


class Agent:
    input_type = "data"

    def __init__(self, input_shapes, output_size, n_symbols, name, role, gibbs_temperature=GIBBS_TEMPERATURE, **kwargs):
        self.name = name
        self.input_shapes = input_shapes
        self.output_size = output_size
        self.n_symbols = n_symbols
        self.role = None
        self.model = None
        self.gibbs_temperature = gibbs_temperature
        self.exploration_rate = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_DECAY
        self.exploration_min = EXPLORATION_FLOOR
        self.set_role(role)

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def set_role(self, role="sender"):
        assert (role in ("sender", "receiver")), role
        self.role = role

    def act(self, state, explore="gibbs", gibbs_temperature=None):
        raise NotImplementedError

    def fit(self, state, action, reward):
        raise NotImplementedError

    def remember(self, state, action, reward):
        raise NotImplementedError

    def batch_train(self):
        raise NotImplementedError

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class MultiAgent(Agent):
    def __init__(self, model_type="old", **kwargs):
        super().__init__(**kwargs)
        self.net = {"sender": component.Net(),
                    "receiver": component.Net()}
        self.model_type = model_type
        if model_type == "old":
            self._build_model(**kwargs)
        elif model_type == "new":
            self._build_model_alt(**kwargs)
        else:
            raise KeyError(f"model_type: '{model_type}")

    def active_net(self):
        assert self.role in self.net.keys()
        return self.net[self.role]

    def act(self, state, explore=False, gibbs_temperature=1.):
        assert explore.lower() in (False, "gibbs", "decay", "false", "none")
        state = [np.expand_dims(st, 0) for st in state]
        act_values = self.active_net().predict(state)
        act_values = np.squeeze(act_values)
        if explore == "gibbs":
            if np.any(np.isnan(act_values)):
                print(act_values)
            if self.model_type == "reinforce":
                probs = softmax(act_values, self.gibbs_temperature)
                action = np.random.choice(range(len(act_values)), 1, p=probs)
            else:
                action = np.random.choice(range(len(act_values)), 1, p=act_values)
        elif explore == "decay":
            if np.random.rand() < self.exploration_rate:
                if self.exploration_rate > self.exploration_min:
                    self.exploration_rate *= self.exploration_rate_decay
                action = np.random.choice(range(len(act_values)), 1)
            else:
                action = np.argmax(act_values)
        else:
            action = np.argmax(act_values)
        # self.last_action = (state, action, act_values)
        return action, act_values

    def remember(self, state, action, reward, net=None):
        if not net:
            self.active_net().remember(state, action, reward)
        else:
            self.net[net].remember(state, action, reward)

    def prepare_batch(self, size: int, **kwargs):
        self.active_net().prepare_batch(size, **kwargs)

    def batch_train(self):
        self.last_loss = self.active_net().batch_train()

    def switch_role(self):
        if self.role == "sender":
            self.set_role("receiver")
        else:
            self.set_role("sender")

    def load(self, name):
        self.net["sender"].model.load_weights(f"{name}.snd")
        self.net["receiver"].model.load_weights(f"{name}.rcv")

    def save(self, name=None):
        if not name:
            time = dt.now().strftime("%y%m%d-%H%M")
            name = f"model-{time}"
        self.net["sender"].model.save_weights(f"{name}.snd")
        self.net["receiver"].model.save_weights(f"{name}.rcv")

    def get_embedding(self, image_vector):
        return self.active_net().embed(image_vector)

    def get_active_name(self):
        return f"{self.name}.{self.role}"

    def _build_model(self, input_shapes, n_symbols, embedding_size=EMBEDDING_SIZE, n_informed_filters=20,
                     use_bias=USE_BIAS, loss=LOSS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     mode="dot", sender_type="agnostic", dropout=0, shared_embedding=True,
                     out_activation="softmax", **kwargs):
        # Shared part
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

        # imgs = [embs[i](inputs[i]) for i in range(n_input_images)]  # separate embedding layer for each image
        imgs = [emb(inputs[i]) for i in range(n_input_images)]  # same embedding layer for all images

        if dropout:
            imgs = [layers.Dropout(dropout)(imgs[i]) for i in range(n_input_images)]

        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                optimizer = optim.Adam
            else:
                raise TypeError(f"Unknown optimizer '{optimizer}'")

        if sender_type == "agnostic":
            out = [layers.Activation("sigmoid")(imgs[i]) for i in range(n_input_images)]
            concat = layers.concatenate(out, axis=-1)
            out = layers.Dense(n_symbols,
                               use_bias=use_bias,
                               )(concat)
        elif sender_type == "informed":
            stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="stack")
            reshape = layers.Reshape((-1, embedding_size, 1))
            feat_filters = layers.Conv2D(filters=n_informed_filters,
                                         kernel_size=(n_input_images, 1),
                                         activation="sigmoid",
                                         data_format="channels_last",
                                         name="feature_filters"
                                         )

            voc_filter = layers.Conv2D(1, (1, n_informed_filters),
                                       activation="linear",
                                       data_format="channels_first",
                                       name="vocab_filter"
                                       )

            dense = layers.Dense(n_symbols, name="output_dense")
            out = dense(layers.Flatten()(voc_filter(feat_filters(reshape(stack(imgs))))))
        else:
            raise KeyError(f"Unknown sender type: {sender_type}")

        # Common sender part
        if self.gibbs_temperature != 0:
            out = layers.Lambda(lambda x: x / self.gibbs_temperature)(out)
        # out = layers.Activation("softmax")(out)
        out = layers.Activation(out_activation)(out)

        aux_input = layers.Input(shape=input_shapes[0])

        self.net["sender"].model = Model(inputs, out)
        self.net["sender"].model.compile(loss=loss, optimizer=optimizer(lr=learning_rate))
        self.net["sender"].emb_output = Model(aux_input, emb(aux_input))
        self.net["sender"].input_shapes = input_shapes[:-1]
        self.net["sender"].output_size = n_symbols
        self.net["sender"].reset_batch()
        self.net["sender"].reset_memory()

        # Receiver part
        symbol_shape = input_shapes[-1]
        sym_input = layers.Input(shape=symbol_shape, dtype="int32", name="input_sym")
        emb_sym = layers.Embedding(input_dim=n_symbols,
                                   output_dim=embedding_size,
                                   name=f"embed_sym")
        symbol = layers.Flatten()(emb_sym(sym_input))
        symbol = layers.Dropout(dropout)(symbol)

        if not shared_embedding:
            emb = layers.Dense(embedding_size,
                               activation='linear',
                               use_bias=use_bias,
                               name=f"embed_img_rcv")
            imgs = [emb(inputs[i]) for i in range(n_input_images)]

        # mode = "dense"
        if mode == "dot":
            dot = layers.Dot(axes=1)
            dot_prods = [dot([img, symbol]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
            # out = layers.Dense(n_input_images)(out)
        elif mode == "dense":
            out = layers.concatenate([*imgs, symbol], axis=-1)
            out = layers.Dense(n_input_images,
                               # activation="sigmoid",
                               name=f"dense_join")(out)
        else:
            raise ValueError(f"'{mode}' is not a valid mode.")

        if self.gibbs_temperature != 0:
            out = layers.Lambda(lambda x: x / self.gibbs_temperature)(out)
        # out = layers.Activation("softmax")(out)
        out = layers.Activation(out_activation)(out)

        aux_input = layers.Input(shape=input_shapes[0])

        self.net["receiver"].model = Model([*inputs, sym_input], out)
        self.net["receiver"].model.compile(loss=loss, optimizer=optimizer(lr=learning_rate))
        self.net["receiver"].emb_output = Model(aux_input, emb(aux_input))
        self.net["receiver"].input_shapes = input_shapes
        self.net["receiver"].output_size = n_input_images
        self.net["receiver"].reset_batch()
        self.net["receiver"].reset_memory()

    def _build_model_alt(self, input_shapes, n_symbols, embedding_size=EMBEDDING_SIZE, n_informed_filters=20,
                     use_bias=USE_BIAS, loss=LOSS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     mode="dot", sender_type="agnostic", dropout=0, shared_embedding=True,
                     out_activation="softmax", **kwargs):
        # Shared part
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

        # imgs = [embs[i](inputs[i]) for i in range(n_input_images)]  # separate embedding layer for each image
        imgs = [emb(inputs[i]) for i in range(n_input_images)]  # same embedding layer for all images

        if dropout:
            imgs = [layers.Dropout(dropout)(imgs[i]) for i in range(n_input_images)]

        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                optimizer = optim.Adam
            else:
                raise TypeError(f"Unknown optimizer '{optimizer}'")

        if sender_type == "agnostic":
            out = [layers.Activation("sigmoid")(imgs[i]) for i in range(n_input_images)]
            concat = layers.concatenate(out, axis=-1)
            out = layers.Dense(n_symbols,
                               use_bias=use_bias,
                               )(concat)
        elif sender_type == "informed":
            stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="stack")
            reshape = layers.Reshape((-1, embedding_size, 1))
            feat_filters = layers.Conv2D(filters=n_informed_filters,
                                         kernel_size=(n_input_images, 1),
                                         activation="sigmoid",
                                         data_format="channels_last",
                                         name="feature_filters"
                                         )

            voc_filter = layers.Conv2D(1, (1, n_informed_filters),
                                       activation="linear",
                                       data_format="channels_first",
                                       name="vocab_filter"
                                       )

            dense = layers.Dense(n_symbols, name="output_dense")
            out = dense(layers.Flatten()(voc_filter(feat_filters(reshape(stack(imgs))))))
        else:
            raise KeyError(f"Unknown sender type: {sender_type}")

        # Common sender part
        if self.gibbs_temperature != 0:
            out = layers.Lambda(lambda x: x / self.gibbs_temperature)(out)
        out = layers.Activation("sigmoid")(out)
        # out = layers.Activation(out_activation)(out)

        aux_input = layers.Input(shape=input_shapes[0])

        self.net["sender"].model = Model(inputs, out)
        self.net["sender"].model.compile(loss="categorical_crossentropy", optimizer=optimizer(lr=learning_rate))
        self.net["sender"].emb_output = Model(aux_input, emb(aux_input))
        self.net["sender"].input_shapes = input_shapes[:-1]
        self.net["sender"].output_size = n_symbols
        self.net["sender"].reset_batch()
        self.net["sender"].reset_memory()

        # Receiver part
        symbol_shape = input_shapes[-1]
        sym_input = layers.Input(shape=symbol_shape, dtype="int32", name="input_sym")
        emb_sym = layers.Embedding(input_dim=n_symbols,
                                   output_dim=embedding_size,
                                   name=f"embed_sym")
        symbol = layers.Flatten()(emb_sym(sym_input))
        symbol = layers.Dropout(dropout)(symbol)

        if not shared_embedding:
            emb = layers.Dense(embedding_size,
                               activation='linear',
                               use_bias=use_bias,
                               name=f"embed_img_rcv")
            imgs = [emb(inputs[i]) for i in range(n_input_images)]

        # mode = "dense"
        if mode == "dot":
            dot = layers.Dot(axes=1)
            dot_prods = [dot([img, symbol]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
            # out = layers.Dense(n_input_images)(out)
        elif mode == "dense":
            out = layers.concatenate([*imgs, symbol], axis=-1)
            out = layers.Dense(n_input_images,
                               # activation="sigmoid",
                               name=f"dense_join")(out)
        else:
            raise ValueError(f"'{mode}' is not a valid mode.")

        if self.gibbs_temperature != 0:
            out = layers.Lambda(lambda x: x / self.gibbs_temperature)(out)
        out = layers.Activation("softmax")(out)
        # out = layers.Activation(out_activation)(out)

        aux_input = layers.Input(shape=input_shapes[0])

        self.net["receiver"].model = Model([*inputs, sym_input], out)
        self.net["receiver"].model.compile(loss="categorical_crossentropy", optimizer=optimizer(lr=learning_rate))
        self.net["receiver"].emb_output = Model(aux_input, emb(aux_input))
        self.net["receiver"].input_shapes = input_shapes
        self.net["receiver"].output_size = n_input_images
        self.net["receiver"].reset_batch()
        self.net["receiver"].reset_memory()


class MultiAgentReinforce(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = {"sender": component.Net(),
                    "receiver": component.Net()}
        self._build_model(**kwargs)

    def active_net(self):
        assert self.role in self.net.keys()
        return self.net[self.role]

    def act(self, state, explore="gibbs", gibbs_temperature=1.):
        state = [np.expand_dims(st, 0) for st in state]
        act_values = self.active_net().predict(state)
        act_values = np.squeeze(act_values)
        if np.any(np.isnan(act_values)):
            print(act_values)
        probs = act_values
        action = np.random.choice(range(len(act_values)), 1, p=probs)
        # self.last_action = (state, action, act_values)
        return action, act_values

    def remember(self, state, action, reward, net=None):
        if not net:
            self.active_net().remember(state, action, reward)
        else:
            self.net[net].remember(state, action, reward)

    def prepare_batch(self, size: int, **kwargs):
        self.active_net().prepare_batch(size, **kwargs)

    def batch_train(self):
        self.last_loss = self.active_net().batch_train()

    def switch_role(self):
        if self.role == "sender":
            self.set_role("receiver")
        else:
            self.set_role("sender")

    def load(self, name):
        self.net["sender"].model.load_weights(f"{name}.snd")
        self.net["receiver"].model.load_weights(f"{name}.rcv")

    def save(self, name=None):
        if not name:
            time = dt.now().strftime("%y%m%d-%H%M")
            name = f"model-{time}"
        self.net["sender"].model.save_weights(f"{name}.snd")
        self.net["receiver"].model.save_weights(f"{name}.rcv")

    def get_embedding(self, image_vector):
        return self.active_net().embed(image_vector)

    def get_active_name(self):
        return f"{self.name}.{self.role}"

    def _build_model(self, input_shapes, n_symbols, embedding_size=EMBEDDING_SIZE, n_informed_filters=20,
                               optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                               sender_type="agnostic", shared_embedding=True, **kwargs):
        print("-!- Using REINFORCE model -!-")
        print("Following user settings will be ignored:")
        print("out_activation, sender_type, optimizer, dropout, shared_embedding")

        # optimizer = optim.Adam
        optimizer = optim.SGD
        temperature = 10
        learning_rate = 0.5

        # Shared part
        n_input_images = len(input_shapes) - 1
        inputs = [layers.Input(shape=input_shapes[i],
                               name=f"input_{i}")
                  for i in range(n_input_images)]
        emb = layers.Dense(embedding_size,
                           activation='linear',
                           name=f"embed_img")

        imgs = [emb(inputs[i]) for i in range(n_input_images)]

        out = [layers.Activation("sigmoid")(imgs[i]) for i in range(n_input_images)]
        concat = layers.concatenate(out, axis=-1)
        out = layers.Dense(n_symbols)(concat)

        temp = layers.Lambda(lambda x: x / temperature)
        soft = layers.Softmax()

        out = soft(temp(out))

        reward = layers.Input((1,), name="reward")

        def custom_loss(y_true, y_pred):
            return - K.mean(K.log(y_pred) * y_true, axis=1) * reward

        self.net["sender"].model_predict = Model(inputs, out)
        self.net["sender"].model_train = Model([*inputs, reward], out)
        # self.net["sender"].model_train = Model([*inputs, reward], out)
        self.net["sender"].model_train.compile(loss=custom_loss, optimizer=optimizer(lr=learning_rate))
        self.net["sender"].input_shapes = input_shapes[:-1]
        self.net["sender"].output_size = n_symbols
        self.net["sender"].reset_batch()
        self.net["sender"].reset_memory()

        ###############
        # Receiver part

        symbol_shape = input_shapes[-1]
        sym_input = layers.Input(shape=symbol_shape, dtype="int32", name="input_sym")
        emb_sym = layers.Embedding(input_dim=n_symbols,
                                   output_dim=embedding_size,
                                   name=f"embed_sym")
        symbol = layers.Flatten()(emb_sym(sym_input))

        shared_embedding = False
        if not shared_embedding:
            emb = layers.Dense(embedding_size,
                               activation='linear',
                               name=f"embed_img")
            imgs = [emb(inputs[i]) for i in range(n_input_images)]  # same embedding layer for all images

        rcv_mode = "dot_product"
        # rcv_mode = "euclidean_distance"
        # rcv_mode = "cosine_similarity"

        if rcv_mode == "dot_product":
            dot = layers.Dot(axes=1)
            dot_prods = [dot([img, symbol]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
            # squeeze the output within (0, 1)
            # out = layers.Activation("sigmoid")(out)
        elif rcv_mode == "euclidean_distance":
            out = [layers.Subtract()([img, symbol]) for img in imgs]
            out = layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x))))(out)
            # squeeze the output within (0, 1)
            # out = layers.Activation("sigmoid")(-1 * out)
        elif rcv_mode == "cosine_similarity":
            raise NotImplementedError(f"rcv_mode: '{rcv_mode}'")
        else:
            raise KeyError(f"rcv_mode: '{rcv_mode}'")

        temp = layers.Lambda(lambda x: x/temperature)
        soft = layers.Softmax()

        out = soft(temp(out))

        self.net["receiver"].model_train = Model([*inputs, sym_input, reward], out)
        self.net["receiver"].model_train.compile(loss=custom_loss, optimizer=optimizer(lr=learning_rate))
        self.net["receiver"].model_predict = Model([*inputs, sym_input], out)
        self.net["receiver"].input_shapes = input_shapes
        self.net["receiver"].output_size = n_input_images
        self.net["receiver"].reset_batch()
        self.net["receiver"].reset_memory()
