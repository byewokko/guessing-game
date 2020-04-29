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
        self.train_model = None
        self.predict_model = None
        self.learning_rate = learning_rate
        self.gibbs_temp = gibbs_temp
        self.exploration_rate = 1.
        self.exploration_rate_decay = 0.99
        self.exploration_min = 0.015
        self.optimizer = optimizer(self.learning_rate, clipnorm=1.)
        self.use_bias = use_bias
        self.batch_states = None
        self.batch_actions = None
        self.batch_rewards = None
        self.reset_batch()

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state, explore=True):
        state = [np.expand_dims(st, 0) for st in state]
        act_probs = self.predict_model.predict(state)
        # act_probs = self.train_model.predict([*state, np.asarray([0])])
        act_probs = np.squeeze(act_probs)
        # TODO: fix the sampling, this is stupid
        if explore and np.random.rand() > self.exploration_rate:
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_rate_decay
            # Sample from Gibbs distribution
            act_probs_exp = np.exp(act_probs / self.gibbs_temp)
            act_probs = act_probs_exp / act_probs_exp.sum()
            # action = np.random.choice(range(self.output_size), 1, p=act_probs)
        else:
            action = np.argmax(act_probs)
        self.last_action = (state, action, act_probs)
        return action, act_probs

    def fit(self, state, action, reward):
        self.last_weights = self.train_model.get_weights()
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(action_onehot, 0)
        self.last_loss = self.train_model.train_on_batch([*X, reward], Y)
        # self.last_updates = self.optimizer.get_updates() # needs args: loss and params

    def remember(self, state, action, reward):
        for i in range(len(state)):
            self.batch_states[i].append(state[i])
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        self.batch_actions.append(action_onehot)
        self.batch_rewards.append(reward)

    def batch_train(self):
        self.last_loss = self.train_model.train_on_batch(
            [np.stack(stack) for stack in [*self.batch_states, self.batch_rewards]],
            np.stack(self.batch_actions)
        )
        self.reset_batch()

    def reset_batch(self):
        self.batch_states = [[] for _ in self.input_sizes]
        self.batch_actions = []
        self.batch_rewards = []

    def load(self, name):
        self.train_model.load_weights(name)

    def save(self, name):
        self.train_model.save_weights(name)


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
                             activation='sigmoid',
                             use_bias=self.use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        emb = layers.Dense(self.embedding_size,
                           activation='sigmoid',
                           use_bias=self.use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42),
                           # kernel_regularizer=l2(0.001),
                           name=f"embed_img")

        imgs = [embs[i](inputs[i]) for i in range(n_inputs)]  # separate embedding layer for each image
        # imgs = [emb(inputs[i]) for i in range(n_inputs)]  # same embedding layer for all images

        concat = layers.concatenate(imgs, axis=-1)
        out = layers.Dense(self.output_size,
                           activation='linear',
                           use_bias=self.use_bias,
                           # kernel_initializer=init.glorot_uniform(seed=42)
                           )(concat)

        temp = layers.Lambda(lambda x: x / self.gibbs_temp)
        soft = layers.Activation("softmax")
        predict_out = soft(temp(out))
        train_out = soft(out)
        # train_out = out

        reward = layers.Input((1,), name="reward")

        def custom_loss(y_true, y_pred):
            # Cross-entropy 1 (??)
            # log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            # return K.mean(log_lik * reward, keepdims=True)

            # Cross-entropy 2
            return K.sum(K.log(y_pred) * y_true) * reward

            # RMS loss
            # return K.mean((K.square(y_pred - y_true))) * reward

        self.train_model = Model([*inputs, reward], train_out)
        self.train_model.compile(loss=custom_loss, optimizer=self.optimizer)
        self.predict_model = Model(inputs, predict_out)
        # self.predict_model.compile(loss=custom_loss, optimizer=self.optimizer)


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
                             activation='sigmoid',
                             use_bias=self.use_bias,
                             # kernel_initializer=init.glorot_uniform(seed=42),
                             # kernel_regularizer=l2(0.001),
                             name=f"embed_{i}")
                for i in range(n_inputs)]

        emb = layers.Dense(self.embedding_size,
                           activation='sigmoid',
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
                                     # padding="same",
                                     # strides=embedding_size,
                                     # data_format="channels_last",
                                     name="feature_filters"
                                     )

        voc_filter = layers.Conv2D(1, (1, n_filters),
                                   # padding="same",
                                   data_format="channels_first",
                                   name="vocab_filter"
                                   )

        dense = layers.Dense(output_size, name="output_dense")

        out = dense(layers.Flatten()(voc_filter(feat_filters(reshape(stack(imgs))))))

        temp = layers.Lambda(lambda x: x / 10, name="gibbs_temp")
        soft = layers.Activation("softmax", name="softmax")
        predict_out = soft(temp(out))
        train_out = soft(out)

        reward = layers.Input((1,), name="reward")

        def custom_loss(y_true, y_pred):
            # Cross-entropy 1 (??)
            # log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            # return K.mean(log_lik * reward, keepdims=True)

            # Cross-entropy 2
            return K.sum(K.log(y_pred) * y_true) * reward

            # RMS loss
            # return K.mean((K.square(y_pred - y_true))) * reward

        self.train_model = Model([*inputs, reward], train_out)
        self.train_model.compile(loss=custom_loss, optimizer=self.optimizer)
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
            # out = layers.Activation("sigmoid")(out)
        elif self.mode == "cosine":
            # basically normalized dot product
            norm = layers.Lambda(lambda v: K.l2_normalize(v))
            dot = layers.Dot(axes=1)
            dot_prods = [dot([norm(img), norm(symbol)]) for img in imgs]
            out = layers.concatenate(dot_prods, axis=-1)
        elif self.mode == "dense":
            dense_join = layers.Dense(self.embedding_size,
                                      activation="sigmoid",
                                      # kernel_initializer=init.glorot_uniform(seed=42),
                                      name=f"dense_join")
            dense_out = layers.Dense(self.output_size,
                                     # kernel_initializer=init.glorot_uniform(seed=42),
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
        # train_out = out

        reward = layers.Input((1,), name="reward")

        def custom_loss(y_true, y_pred):
            # Cross-entropy 1 (??)
            # log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            # return K.mean(log_lik * reward, keepdims=True)

            # Cross-entropy 2
            return K.sum(K.log(y_pred) * y_true) * reward

            # RMS loss
            # return K.mean((K.square(y_pred - y_true))) * reward

        self.train_model = Model([*inputs, sym_input, reward], train_out)
        self.train_model.compile(loss=custom_loss, optimizer=self.optimizer)
        self.predict_model = Model([*inputs, sym_input], predict_out)
