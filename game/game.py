import numpy as np
import logging

REWARD = {
    "sender_success": 1,
    "receiver_success": 1,
    "sender_fail": 0,
    "receiver_fail": 0
}

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.propagate = True


class Game:

    image_dict: dict
    reward: dict

    def __init__(self, images, images_filenames=None, images_framed=None, sender=None, receiver=None, reward=None):
        self.images = images
        self.images_filenames = images_filenames
        self.images_framed = images_framed
        self.use_frame = images_framed is not None
        self.image_ids = np.arange(len(self.images))
        self.sender = sender
        self.receiver = receiver
        self.reward = reward or REWARD
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
        receiver_state = [sender_action, *receiver_images]
        receiver_action, receiver_prob = self.receiver.act(receiver_state)
        log.debug(f"Guess: {sample_ids[receiver_action]} at p{receiver_action}. Correct is {correct_id} at p{correct_pos}.")

        # Evaluate and reward
        if receiver_action == correct_pos:
            #self.sender.reward_send(self.reward["sender_success"])
            #self.receiver.reward_receive(self.reward["receiver_success"])
            # self.sender.reinforce(sender_state, sender_action, sender_prob, self.reward["sender_success"])
            # self.receiver.reinforce(receiver_state, receiver_action, receiver_prob, self.reward["receiver_success"])
            self.sender.fit(sender_state, sender_action, self.reward["sender_success"])
            self.receiver.fit(receiver_state, receiver_action, self.reward["receiver_success"])
            log.debug("Correct")
            is_success = True
        else:
            # self.sender.reward_send(self.reward["sender_fail"])
            # self.sender.reinforce(sender_state, sender_action, sender_prob, self.reward["sender_fail"])
            # self.receiver.reinforce(receiver_state, receiver_action, receiver_prob, self.reward["receiver_fail"])
            self.sender.fit(sender_state, sender_action, self.reward["sender_fail"])
            self.receiver.fit(receiver_state, receiver_action, self.reward["receiver_fail"])
            log.debug("Wrong")
            is_success = False
        log.info(f"Turn {self.episode} finished: {'SUCCESS' if is_success else 'FAIL'}.")
        self.episode += 1
        return is_success

    def switch_roles(self):
        tmp = self.sender
        self.sender = self.receiver
        self.receiver = tmp
        log.debug(f"Roles switched")
