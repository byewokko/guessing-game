import numpy as np
from keras.models import Model
import keras.layers as layers
import keras.optimizers as optim
import keras.backend as K
import keras.initializers as init
from keras.losses import categorical_crossentropy
from keras.regularizers import l2

from tools.debug import print_layer


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.mean(x * y, axis=-1, keepdims=True)


class Agent:
    input_type = "data"

    def __init__(self, input_sizes, output_size, n_symbols,
                 embedding_size=50, learning_rate=0.001, gibbs_temp=10,
                 optimizer=optim.Adam, use_bias=True,
                 **kwargs):
        self.n_input_images = 2
        self.input_sizes = input_sizes
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_symbols = n_symbols
        self.last_action = None
        self.last_loss = 0
        self.train_model = None
        self.predict_model = None
        self.learning_rate = learning_rate
        self.gibbs_temp = gibbs_temp
        self.optimizer = optimizer
        self.use_bias = use_bias

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state):
        state = [np.expand_dims(st, 0) for st in state]
        act_probs = self.predict_model.predict(state)
        act_probs = np.squeeze(act_probs)
        action = np.random.choice(range(self.output_size), 1, p=act_probs)
        self.last_action = (state, action, act_probs)
        return action, act_probs

    def fit(self, state, action, reward):
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(action_onehot, 0)
        self.last_loss = self.train_model.train_on_batch([*X, reward], Y)

    def load(self, name):
        self.train_model.load_weights(name)

    def save(self, name):
        self.train_model.save_weights(name)


class Sender(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_model()

    def _build_model(self, **kwargs):
        n_inputs = len(self.input_sizes)
        inputs = [layers.Input(shape=self.input_sizes[i],
                               name=f"input_{i}")
                  for i in range(n_inputs)]
        embs = [layers.Dense(self.embedding_size,
                             activation='sigmoid',
                             use_bias=self.use_bias,
                             kernel_initializer=init.glorot_uniform(seed=42),
                             # activity_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        emb = layers.Dense(self.embedding_size,
                           activation='sigmoid',
                           use_bias=self.use_bias,
                           kernel_initializer=init.glorot_uniform(seed=42),
                           name=f"embed_img")

        imgs = [embs[i](inputs[i]) for i in range(n_inputs)]  # separate embedding layer for each image
        # imgs = [emb(inputs[i]) for i in range(n_inputs)]  # same embedding layer for all images

        concat = layers.concatenate(imgs, axis=-1)
        out = layers.Dense(self.output_size,
                           activation='linear',
                           use_bias=self.use_bias,
                           kernel_initializer=init.glorot_uniform(seed=42))(concat)

        temp = layers.Lambda(lambda x: x / self.gibbs_temp)
        soft = layers.Activation("softmax")
        predict_out = soft(temp(out))
        train_out = soft(out)

        reward = layers.Input((1,), name="reward")

        def custom_loss(y_true, y_pred):
            # log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            # return K.mean(log_lik * reward, keepdims=True)
            log_lik = K.sum(K.log(y_pred) * y_true)
            return log_lik * reward

        self.train_model = Model([*inputs, reward], train_out)
        self.train_model.compile(loss=custom_loss, optimizer=self.optimizer(self.learning_rate))
        self.predict_model = Model(inputs, predict_out)


class Receiver(Agent):
    def __init__(self, mode="dot", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self._build_model()

    def _build_model(self, **kwargs):
        n_input_images = len(self.input_sizes) - 1
        inputs = [layers.Input(shape=self.input_sizes[i],
                               name=f"input_{i}")
                  for i in range(n_input_images)]
        embs = [layers.Dense(self.embedding_size,
                             activation='linear',
                             use_bias=self.use_bias,
                             kernel_initializer=init.glorot_uniform(seed=42),
                             # activity_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_input_images)]
        emb = layers.Dense(self.embedding_size,
                           activation='linear',
                           use_bias=self.use_bias,
                           kernel_initializer=init.glorot_uniform(seed=42),
                           # activity_regularizer=l2(0.001),
                           name=f"embed_img")

        # imgs = [embs[i](inputs[i]) for i in range(n_input_images)]  # separate embedding layer for each image
        imgs = [emb(inputs[i]) for i in range(n_input_images)]  # same embedding layer for all images

        symbol_shape = self.input_sizes[-1]
        sym_input = layers.Input(shape=symbol_shape, dtype="int32", name="input_sym")
        emb_sym = layers.Embedding(input_dim=self.n_symbols,
                                   output_dim=self.embedding_size,
                                   name=f"embed_sym")
        flat = layers.Flatten()
        symbol = flat(emb_sym(sym_input))

        if self.mode == "dot":
            dot = layers.Dot(axes=1)
            dot_prods = [dot([img, symbol]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
        elif self.mode == "dense":
            dense_join = layers.Dense(self.embedding_size,
                                      activation="sigmoid",
                                      kernel_initializer=init.glorot_uniform(seed=42),
                                      name=f"dense_join")
            dense_out = layers.Dense(self.output_size,
                                     kernel_initializer=init.glorot_uniform(seed=42),
                                     name=f"dense_out")
            out = dense_out(dense_join(layers.concatenate([*imgs, symbol], axis=-1)))
        else:
            raise ValueError(f"'{self.mode}' is not a valid mode.")

        # cossim = layers.Lambda(cosine_distance, output_shape=(1,))
        # sims = [cossim([img, symbol]) for img in imgs]
        # out = layers.concatenate(sims, axis=-1)
        # p = print_layer(out, "rec 1")
        temp = layers.Lambda(lambda x: x / self.gibbs_temp)
        soft = layers.Activation("softmax")
        predict_out = soft(temp(out))
        train_out = soft(out)

        reward = layers.Input((1,), name="reward")

        def custom_loss(y_true, y_pred):
            # log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            # return K.mean(log_lik * reward, keepdims=True)
            log_lik = K.sum(K.log(y_pred) * y_true)
            return log_lik * reward

        self.train_model = Model([*inputs, sym_input, reward], train_out)
        self.train_model.compile(loss=custom_loss, optimizer=self.optimizer(self.learning_rate))
        self.predict_model = Model([*inputs, sym_input], predict_out)
