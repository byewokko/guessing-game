import numpy as np
import logging

REWARD = {
    "success": 1.,
    "fail": 0.
}

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.propagate = True


def np_reward(reward):
    if reward:
        return {k: np.asarray([v]) for k, v in reward.items()}
    return None


class Game:

    image_dict: dict
    reward: dict

    def __init__(self, images, categories=None, images_filenames=None, images_framed=None,
                 reward=None, reward_sender=None, reward_receiver=None):
        self.images = images
        self.images_filenames = images_filenames
        self.images_framed = images_framed
        self.use_frame = images_framed is not None
        self.image_ids = np.arange(len(self.images), dtype="int32")
        self.reward = np_reward(reward) or np_reward(REWARD)
        self.reward_sender = np_reward(reward_sender) or self.reward
        self.reward_receiver = np_reward(reward_receiver) or self.reward
        self.episode = 0

        self.categories = None
        if categories is not None:
            categories = np.asarray(categories)
            self.categories = np.unique(categories)
            self.categorized_images = {}
            for c in self.categories:
                idx = np.where(categories == c)
                self.categorized_images[c] = self.image_ids[idx]

        self.sample_ids = None
        self.sample_ctgs = None
        self.correct_pos_sender = None
        self.correct_pos_receiver = None
        self.correct_id = None

    def take_turn(self, sender, receiver, n_images=2):
        self.reset()
        sender_state = self.get_sender_state(n_images)
        sender_action, sender_prob = sender.act(sender_state)
        receiver_state = self.get_receiver_state(sender_action)
        receiver_action, receiver_prob = receiver.act(receiver_state)
        sender_reward, receiver_reward, success = self.evaluate_guess(receiver_action)
        sender.remember(sender_state, sender_action, sender_reward)
        receiver.remember(receiver_state, receiver_action, receiver_reward)
        self.episode += 1
        return success

    def generate_games(self, n_games, n_images=2, game_type="different_synset"):
        games = []
        for g in range(n_games):
            if game_type == "different_synset":
                sample_ctgs = np.random.choice(a=self.categories, size=n_images, replace=False)
                sender_ids = np.zeros(n_images, dtype="int32")
                for i in range(n_images):
                    sender_ids[i] = np.random.choice(self.categorized_images[sample_ctgs[i]], size=1)
            elif game_type == "same_synset":
                sample_ctg = np.random.choice(a=self.categories)
                sender_ids = np.random.choice(self.categorized_images[sample_ctg], size=n_images, replace=False)
            else:
                sender_ids = np.random.choice(a=self.image_ids, size=n_images, replace=False)
            correct_id = sender_ids[0]

            receiver_ids = np.copy(sender_ids)
            np.random.shuffle(receiver_ids)

            correct_pos_receiver = np.where(receiver_ids == correct_id)[0]

            games.append({
                "sender_ids": sender_ids,
                "receiver_ids": receiver_ids,
                "receiver_pos": correct_pos_receiver,
            })
        return games

    def get_sender_state_from_ids(self, ids, expand=True):
        sender_state = self.images[ids]
        if expand:
            sender_state = [np.expand_dims(item, axis=0) for item in sender_state]
        return sender_state

    def get_receiver_state_from_ids(self, ids, correct_pos, sender_action, expand=True):
        self.correct_pos_receiver = correct_pos
        receiver_images = self.images[ids]
        if expand:
            sender_action = np.asarray([sender_action])
        receiver_state = [*receiver_images, sender_action]

        if expand:
            receiver_state = [np.expand_dims(item, axis=0) for item in receiver_state]
        return receiver_state

    def get_sender_state(self, n_images=2, unique_categories=True, return_ids=False, expand=True):
        if unique_categories and self.categories is not None:
            self.sample_ctgs = np.random.choice(a=self.categories, size=n_images, replace=False)
            self.sample_ids = np.zeros(n_images, dtype="int32")
            for i in range(n_images):
                self.sample_ids[i] = np.random.choice(self.categorized_images[self.sample_ctgs[i]], size=1)
        else:
            self.sample_ids = np.random.choice(a=self.image_ids, size=n_images, replace=False)
        self.correct_pos_sender = 0
        self.correct_id = self.sample_ids[self.correct_pos_sender]
        log.debug(f"Picked images {self.sample_ids}. Correct {self.correct_id}.")
        sender_state = self.images[self.sample_ids]
        if expand:
            sender_state = [np.expand_dims(item, axis=0) for item in sender_state]

        if return_ids:
            return sender_state, self.sample_ids
        return sender_state

    def get_receiver_state(self, sender_action, return_ids=False, expand=False):
        # Prepare receiver setup
        np.random.shuffle(self.sample_ids)
        receiver_images = self.images[self.sample_ids]
        self.correct_pos_receiver = np.where(self.sample_ids == self.correct_id)[0]

        log.debug("Receiving... ")
        if expand:
            sender_action = np.asarray([sender_action])
        receiver_state = [*receiver_images, sender_action]
        # receiver_state = [*receiver_images, self.correct_pos_receiver]  # dummy state

        if expand:
            receiver_state = [np.expand_dims(item, axis=0) for item in receiver_state]

        if return_ids:
            return receiver_state, self.sample_ids
        return receiver_state

    def evaluate_guess(self, receiver_action):
        if np.all(receiver_action == self.correct_pos_receiver):
            sender_r = self.reward_sender["success"]
            receiver_r = self.reward_receiver["success"]
            is_success = True
        else:
            sender_r = self.reward_sender["fail"]
            receiver_r = self.reward_receiver["fail"]
            is_success = False
        log.info(f"Turn {self.episode} finished: {'SUCCESS' if is_success else 'FAIL'}.")
        self.episode += 1
        return sender_r, receiver_r, is_success

    def reset(self):
        self.sample_ids = None
        self.sample_ctgs = None
        self.correct_pos_sender = None
        self.correct_pos_receiver = None
        self.correct_id = None

    def switch_roles(self):
        tmp = self.sender
        self.sender = self.receiver
        self.receiver = tmp
        log.debug(f"Roles switched")

    def get_sample_ids(self):
        return self.sample_ids
