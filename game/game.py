import numpy as np
import logging

REWARD = {
    "success": 1,
    "fail": 0
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
            self.categories = np.unique(categories)
            self.categorized_images = {}
            for c in self.categories:
                idx = np.where(categories == c)
                self.categorized_images[c] = self.image_ids[idx]

        self.sample_ids = None
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

    def get_sender_state(self, n_images=2, unique_categories=False):
        if unique_categories:
            self.sample_ids = np.zeros(n_images, dtype="int32")
            sample_ctgs = np.random.choice(a=self.categories, size=n_images, replace=False)
            for i in range(n_images):
                self.sample_ids[i] = np.random.choice(self.categorized_images[sample_ctgs[i]], size=1)
        else:
            self.sample_ids = np.random.choice(a=self.image_ids, size=n_images, replace=False)
        self.correct_pos_sender = 0
        self.correct_id = self.sample_ids[self.correct_pos_sender]
        log.debug(f"Picked images {self.sample_ids}. Correct {self.correct_id}.")
        sender_state = self.images[self.sample_ids]

        # Prepare receiver setup
        np.random.shuffle(self.sample_ids)
        receiver_images = self.images[self.sample_ids]
        self.correct_pos_receiver = np.where(self.sample_ids == self.correct_id)[0]

        return sender_state

    def get_receiver_state(self, sender_action):
        receiver_images = self.images[self.sample_ids]
        log.debug("Receiving... ")
        receiver_state = [*receiver_images, sender_action]
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
