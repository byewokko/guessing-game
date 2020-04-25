import numpy as np
from keras.models import Model
import keras.layers as layers
import keras.optimizers as optim
import keras.backend as K
import keras.initializers as init
from keras.losses import categorical_crossentropy
from keras.regularizers import l2

from utils.debug import print_layer


def cosine_distance(stack):
    x, y = stack
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.dot(x, y)


class Agent:
    input_type = "data"

    def __init__(self, input_sizes, output_size, n_symbols,
                 embedding_size=50, learning_rate=0.001, gibbs_temp=10,
                 optimizer=optim.Adam, use_bias=True,
                 **kwargs):
        self.last_updates = None
        self.n_input_images = 2
        self.input_sizes = input_sizes
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_symbols = n_symbols
        self.last_action = None
        self.last_loss = 0
        self.last_weights = None
        self.model = None
        self.exploration_rate = 1.
        self.exploration_rate_decay = 0.99
        self.exploration_min = 0.015
        self.learning_rate = learning_rate
        self.gibbs_temp = gibbs_temp
        self.optimizer = optimizer(self.learning_rate)
        self.use_bias = use_bias
        self.batch_states = None
        self.batch_actions = None
        self.batch_rewards = None
        self.reset_batch()

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state, explore=True):
        state = [np.expand_dims(st, 0) for st in state]
        #action = np.zeros(self.output_size)
        act_probs = np.zeros(self.output_size)
        if explore and np.random.rand() > self.exploration_rate:
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_rate_decay
            action = np.random.randint(0, self.output_size)
        else:
            act_probs = self.model.predict(state)
            # act_probs = self.train_model.predict([*state, np.asarray([0])])
            act_probs = np.squeeze(act_probs)
            action = np.argmax(act_probs)
            # action = np.random.choice(range(self.output_size), 1, p=act_probs)
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
        self.batch_states = [[] for _ in self.input_sizes]
        self.batch_actions = []
        self.batch_rewards = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Sender(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        n_inputs = len(self.input_sizes)
        inputs = [layers.Input(shape=self.input_sizes[i],
                               name=f"input_{i}")
                  for i in range(n_inputs)]
        embs = [layers.Dense(self.embedding_size,
                             activation="linear",
                             use_bias=self.use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        imgs = [embs[i](inputs[i]) for i in range(n_inputs)]  # separate embedding layer for each image

        concat = layers.concatenate(imgs, axis=-1)
        # hidden = layers.Dense(self.embedding_size,
        #                    activation='relu',
        #                    use_bias=self.use_bias,
        #                    # kernel_initializer=init.glorot_uniform(seed=42)
        #                    )(concat)
        out = layers.Dense(self.output_size,
                           activation='sigmoid',
                           use_bias=self.use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42)
                           )(concat)

        self.model = Model(inputs, out)
        self.model.compile(loss="mse", optimizer=self.optimizer)


class SenderInformed(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_model(**kwargs)

    def _build_model(self, input_sizes, embedding_size, n_filters, output_size, use_bias=True, **kwargs):
        n_inputs = len(self.input_sizes)
        inputs = [layers.Input(shape=self.input_sizes[i],
                               name=f"input_{i}")
                  for i in range(n_inputs)]
        embs = [layers.Dense(self.embedding_size,
                             activation='linear',
                             use_bias=self.use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        emb = layers.Dense(self.embedding_size,
                           activation='linear',
                           use_bias=self.use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42),
                           # kernel_regularizer=l2(0.001),
                           name=f"embed_img")

        imgs = [embs[i](inputs[i]) for i in range(n_inputs)]  # separate embedding layer for each image
        # imgs = [emb(inputs[i]) for i in range(n_inputs)]  # same embedding layer for all images

        stack = layers.Lambda(lambda x: K.stack(x, axis=1), name="stack")
        reshape = layers.Reshape((-1, self.embedding_size, 1))
        feat_filters = layers.Conv2D(filters=n_filters,
                                     kernel_size=(2, 1),
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
        self.model.compile(loss="mse", optimizer=self.optimizer)


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
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_input_images)]
        emb = layers.Dense(self.embedding_size,
                           activation='linear',
                           use_bias=self.use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42),
                           # kernel_regularizer=l2(0.001),
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
        elif self.mode == "cosine":
            # basically normalized dot product
            norm = layers.Lambda(lambda v: K.l2_normalize(v))
            dot = layers.Dot(axes=1)
            dot_prods = [dot([norm(img), norm(symbol)]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
        elif self.mode == "dense":
            out = layers.concatenate([*imgs, symbol], axis=-1)
            out = layers.Dense(self.embedding_size,
                               # activation=layers.PReLU(),
                               # kernel_initializer=init.glorot_uniform(seed=42),
                               name=f"dense_join")(out)
            out = layers.Dense(self.output_size,
                               # kernel_initializer=init.glorot_uniform(seed=42),
                               name=f"dense_out")(out)
        else:
            raise ValueError(f"'{self.mode}' is not a valid mode.")

        out = layers.Activation("softmax")(out)

        self.model = Model([*inputs, sym_input], out)
        self.model.compile(loss="mse", optimizer=self.optimizer)
