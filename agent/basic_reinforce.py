import numpy as np
import scipy.stats as stats
from collections import deque
from keras.models import Model
import keras.layers as layers
import keras.optimizers as optim
import keras.backend as K
from keras.losses import categorical_crossentropy


def neg_categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return - K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def custom_ce_loss(reward):
    def custom_loss(y_true, y_pred):
        return -K.mean(reward * y_true * K.log(y_pred), axis=1)
    return custom_loss


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


class Agent:
    """
        Uses feature vectors instead of raw images.
        """
    input_type = "data"

    def __init__(self, n_symbols, n_images, img_shape, embedding_size, memory_size=512, **kwargs):
        self.n_input_images = 2
        self.action_size = None
        self.memory = deque(maxlen=memory_size)
        self.gradients = deque(maxlen=memory_size)
        self.states = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.probs = deque(maxlen=memory_size)
        self.last_action = None
        self.last_loss = 0
        self.model = None
        # self.expl_rate = False
        # self.expl_rate_decay = 0.95
        self.learning_rate = 0.005

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _build_train_fn(self):
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=self.model.output.shape,
                                                  name="action_onehot")
        reward_placeholder = K.placeholder(shape=(1,),
                                           name="reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=-1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * reward_placeholder

        adam = optim.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           reward_placeholder],
                                   outputs=self.model.output,
                                   updates=updates)

    def act(self, state):
        state = [np.expand_dims(st, 0) for st in state]
        act_probs = np.squeeze(self.model.predict(state, batch_size=1))
        self.probs.append(act_probs)
        # norm_probs = act_probs / act_probs.sum()
        # print(act_probs, norm_probs)
        action = np.random.choice(self.action_size, 1, p=act_probs)
        self.last_action = (state, action, act_probs)
        return action, act_probs

    def fit(self, state, action, reward):
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(action_onehot, 0)
        # self.model.compile(optimizer=optim.Adam(lr=self.learning_rate),
        #                    loss=custom_ce_loss(reward))
        self.last_loss = np.mean(self.train_fn([X, Y, reward]))

    def memorize(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def reward(self, reward):
        self.memorize(*self.last_action, reward=reward)

    def test_reinforce(self, state, action, probs, reward):
        y = np.zeros([self.action_size])
        y[action] = reward
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(y.astype('float32'), 0)
        self.last_loss = self.model.train_on_batch(X, Y)

    def old_reinforce(self, state, action, probs, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        gradients = y.astype('float32') - probs
        gradients *= reward
        X = [np.expand_dims(st, 0) for st in state]
        Y = probs + self.learning_rate * gradients
        Y = np.expand_dims(Y, 0)
        self.last_loss = self.model.train_on_batch(X, Y)

    def reinforce(self, state, action, probs, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        gradients = y.astype('float32') * probs
        gradients *= reward
        X = [np.expand_dims(st, 0) for st in state]
        Y = probs + self.learning_rate * gradients
        Y = np.expand_dims(Y, 0)
        self.last_loss = self.model.train_on_batch(X, Y)

    def experience_replay(self, n=10, sampling_mode="norm-tail"):
        raise NotImplementedError()
        batch = sample_memory(self.memory, n, sampling_mode)
        if not batch:
            print("Not enough experience. Skipping training.")
            return
        else:
            for state, action, reward in batch:
                q_values = self.model.predict(state)
                q_values[0][action] = reward
                self.model.fit(state, q_values, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Sender(Agent):
    def __init__(self, n_symbols, n_images, img_shape, embedding_size, **kwargs):
        super().__init__(n_symbols, n_images, img_shape, embedding_size, **kwargs)
        self.action_size = n_symbols
        self._build_model(n_symbols=n_symbols, img_shape=img_shape, embedding_size=embedding_size, **kwargs)
        self._build_train_fn()

    def _build_model(self, img_shape, n_symbols, embedding_size, temperature=10, **kwargs):
        img_input_0 = layers.Input(shape=img_shape, name='image_0')
        img_input_1 = layers.Input(shape=img_shape, name='image_1')
        img_emb_0 = layers.Dense(embedding_size, activation='sigmoid')
        img_emb_1 = layers.Dense(embedding_size, activation='sigmoid')

        img_0 = img_emb_0(img_input_0)
        img_1 = img_emb_1(img_input_1)

        concat = layers.concatenate([img_0, img_1], axis=-1)
        out = layers.Dense(n_symbols)(concat)

        temp = layers.Lambda(lambda x: x / temperature)
        soft = layers.Activation("softmax")
        out = soft(temp(out))

        self.model = Model([img_input_0, img_input_1], out)

        # self.model.compile(optimizer=optim.Adam(lr=self.learning_rate),
        #                    loss=categorical_crossentropy)

    def _build_train_fn(self):
        super()._build_train_fn()


class Receiver(Agent):
    def __init__(self, n_symbols, n_images, img_shape, embedding_size, **kwargs):
        super().__init__(n_symbols, n_images, img_shape, embedding_size, **kwargs)
        self.action_size = n_images
        self._build_model(n_symbols=n_symbols, img_shape=img_shape, embedding_size=embedding_size, **kwargs)
        self._build_train_fn()

    def _build_model(self, img_shape, n_symbols, embedding_size, temperature=10, **kwargs):
        img_input_0 = layers.Input(shape=img_shape, name='image_0')
        img_input_1 = layers.Input(shape=img_shape, name='image_1')
        img_emb_0 = layers.Dense(embedding_size, activation='linear')
        img_emb_1 = layers.Dense(embedding_size, activation='linear')

        img_0 = img_emb_0(img_input_0)
        img_1 = img_emb_1(img_input_1)

        sym_input = layers.Input(shape=(1,), dtype="int32", name="symbol")
        embed = layers.Embedding(input_dim=n_symbols, output_dim=embedding_size)
        resh = layers.Flatten()
        symbol = resh(embed(sym_input))

        dot = layers.Dot(axes=1)
        dot_0 = dot([img_0, symbol])
        dot_1 = dot([img_1, symbol])
        out = layers.concatenate([dot_0, dot_1], axis=-1)

        temp = layers.Lambda(lambda x: x / temperature)
        soft = layers.Activation("softmax")
        out = soft(temp(out))

        self.model = Model([sym_input, img_input_0, img_input_1], out)

        # self.model.compile(optimizer=optim.Adam(lr=self.learning_rate),
        #                    loss=categorical_crossentropy)

    def _build_train_fn(self):
        super()._build_train_fn()

