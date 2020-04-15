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

    def __init__(self, images, images_filenames=None, images_framed=None,
                 sender=None, receiver=None,
                 reward=None, reward_sender=None, reward_receiver=None):
        self.images = images
        self.images_filenames = images_filenames
        self.images_framed = images_framed
        self.use_frame = images_framed is not None
        self.image_ids = np.arange(len(self.images))
        self.sender = sender
        self.receiver = receiver
        self.reward = np_reward(reward) or np_reward(REWARD)
        self.reward_sender = np_reward(reward_sender) or self.reward
        self.reward_receiver = np_reward(reward_receiver) or self.reward
        self.episode = 0

    def take_turn(self, n_images=2):
        log.debug("Turn started")

        # Sender phase
        sample_ids = np.random.choice(a=self.image_ids, size=n_images, replace=False)
        if self.use_frame:
            raise NotImplementedError()
            correct_pos = np.random.randint(0, len(sample_ids))
        else:
            correct_pos = 0
        correct_id = sample_ids[correct_pos]
        log.debug(f"Picked images {sample_ids}. Correct {correct_id}.")

        if self.sender.input_type and self.sender.input_type == "filename":
            raise NotImplementedError()
            sender_state = self.images_filenames[sample_ids]
        else:  # "data"
            sender_state = self.images[sample_ids]
            if self.use_frame:
                raise NotImplementedError()
                sender_state[correct_pos] = self.images_framed[correct_id]

        log.debug("Sending... ")
        sender_action, sender_prob = self.sender.act(sender_state)
        log.debug(f"Clue: {sender_action}")

        # Receiver phase
        np.random.shuffle(sample_ids)
        if self.sender.input_type and self.sender.input_type == "filename":
            receiver_images = self.images_filenames[sample_ids]
        else:
            receiver_images = self.images[sample_ids]
        correct_pos = np.where(sample_ids == correct_id)[0]
        log.debug("Receiving... ")
        receiver_state = [*receiver_images, sender_action]
        receiver_action, receiver_prob = self.receiver.act(receiver_state)
        log.debug(f"Guess: {sample_ids[receiver_action]} at p{receiver_action}. Correct is {correct_id} at p{correct_pos}.")

        # Evaluate and reward
        if receiver_action == correct_pos:
            self.sender.fit(sender_state, sender_action, self.reward_sender["success"])
            self.receiver.fit(receiver_state, receiver_action, self.reward_receiver["success"])
            log.debug("Correct")
            is_success = True
        else:
            self.sender.fit(sender_state, sender_action, self.reward_sender["fail"])
            self.receiver.fit(receiver_state, receiver_action, self.reward_receiver["fail"])
            log.debug("Wrong")
            is_success = False
        log.info(f"Turn {self.episode} finished: {'SUCCESS' if is_success else 'FAIL'}.")
        self.episode += 1
        return is_success

    def play(self, sender, receiver, n_images=2):
        log.debug("Turn started")

        # Sender phase
        sample_ids = np.random.choice(a=self.image_ids, size=n_images, replace=False)
        correct_pos = 0
        correct_id = sample_ids[correct_pos]
        log.debug(f"Picked images {sample_ids}. Correct {correct_id}.")
        sender_state = self.images[sample_ids]

        log.debug("Sending... ")
        sender_action, sender_prob = sender.act(sender_state)
        log.debug(f"Clue: {sender_action}")

        # Receiver phase
        np.random.shuffle(sample_ids)
        receiver_images = self.images[sample_ids]
        correct_pos = np.where(sample_ids == correct_id)[0]
        log.debug("Receiving... ")
        receiver_state = [*receiver_images, sender_action]
        receiver_action, receiver_prob = receiver.act(receiver_state)
        log.debug(
            f"Guess: {sample_ids[receiver_action]} at p{receiver_action}. Correct is {correct_id} at p{correct_pos}.")

        # Evaluate and reward
        if receiver_action == correct_pos:
            sender_sar = (sender_state, sender_action, self.reward_sender["success"])
            receiver_sar = (receiver_state, receiver_action, self.reward_receiver["success"])
            is_success = True
        else:
            sender_sar = (sender_state, sender_action, self.reward_sender["fail"])
            receiver_sar = (receiver_state, receiver_action, self.reward_receiver["fail"])
            is_success = False
        log.info(f"Turn {self.episode} finished: {'SUCCESS' if is_success else 'FAIL'}.")
        self.episode += 1
        return sender_sar, receiver_sar

    def reset(self):
        pass

    def switch_roles(self):
        tmp = self.sender
        self.sender = self.receiver
        self.receiver = tmp
        log.debug(f"Roles switched")
