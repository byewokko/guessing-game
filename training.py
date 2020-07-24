import numpy as np
import game.game as game
import agent.q_agent as agent
import pandas as pd
from utils.dataprep import load_emb_gz, make_categories
from keras.optimizers import Adam, SGD, Adagrad
import matplotlib.pyplot as plt


def run_training(game, agent1, agent2, n_episodes, batch_size, n_images_to_guess_from,
                 roles="switch", show_plot=True, explore="gibbs", gibbs_temperature=0.01, **kwargs):
    if show_plot:
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.65)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        plt.ion()
        fig.show()
        fig.canvas.draw()

    t = []
    show_steps = 2500
    batch_success = []
    success_rate = []
    sendr1_loss = []
    recvr1_loss = []
    sendr2_loss = []
    recvr2_loss = []
    success_rate_avg = []
    sender = agent1
    receiver = agent2
    for i in range(1, n_episodes + 1):
        game.reset()
        sender_state = game.get_sender_state(n_images=n_images_to_guess_from, unique_categories=True)
        sender_action, _ = sender.act(sender_state, explore=explore, gibbs_temperature=gibbs_temperature)
        receiver_state = game.get_receiver_state(sender_action)
        receiver_action, _ = receiver.act(receiver_state, explore=explore, gibbs_temperature=gibbs_temperature)
        sender_reward, receiver_reward, success = game.evaluate_guess(receiver_action)
        sender.remember(sender_state, sender_action, sender_reward)
        receiver.remember(receiver_state, receiver_action, receiver_reward)
        batch_success.append(success)

        if not i % batch_size:
            avg_success = sum(batch_success) / len(batch_success)
            batch_success = []
            sender.batch_train()
            receiver.batch_train()

            if roles == "switch":
                sender.switch_role()
                receiver.switch_role()
                tmp = sender
                sender = receiver
                receiver = tmp

            # PLOT PROGRESS
            t.append(i)
            success_rate.append(avg_success)
            success_rate_avg.append(sum(success_rate[-10:]) / min(10, len(success_rate)))
            sendr1_loss.append(agent1.net["sender"].last_loss)
            sendr2_loss.append(agent2.net["sender"].last_loss)
            recvr1_loss.append(agent1.net["receiver"].last_loss)
            recvr2_loss.append(agent2.net["receiver"].last_loss)
            if not i % (50 * batch_size):
                print(f"Episode {i}")
                print(f"Batch success rate {success_rate_avg[-1]}")
            if not show_plot:
                continue
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax1.set_title("Sender loss")
            ax2.set_title("Receiver loss")
            ax3.set_title("Average batch success rate")
            ax1.plot(t[-show_steps:], sendr1_loss[-show_steps:], "m", label="Agent 1")
            ax1.plot(t[-show_steps:], sendr2_loss[-show_steps:], "c", label="Agent 2")
            ax1.legend(loc="lower left")
            ax2.plot(t[-show_steps:], recvr1_loss[-show_steps:], "m", label="Agent 1")
            ax2.plot(t[-show_steps:], recvr2_loss[-show_steps:], "c", label="Agent 2")
            ax2.legend(loc="lower left")
            ax3.plot(t[-show_steps:], success_rate[-show_steps:], "r.")
            ax3.plot(t[-show_steps:], success_rate_avg[-show_steps:], "k")
            fig.canvas.draw()
            fig.canvas.flush_events()

    print("Training finished")
    # plt.show(block=True)


def run_test(game, agent1, agent2, res_file, n_episodes, batch_size, n_active_images,
             roles="switch", show_plot=False, explore=None, gibbs_temperature=0.01, **kwargs):
    # Set up the logging of results
    columns = "sender_name,receiver_name,active_images,target_image,chosen_symbol,chosen_symbol_p,\
    chosen_image,chosen_image_p,success".replace(" ", "").split(",")
    results = {k: None for k in columns}
    df_results = pd.DataFrame(columns=columns)
    # df_results.set_index("episode", inplace=True)

    # Progress plotting set up
    if show_plot:
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.65)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        plt.ion()
        fig.show()
        fig.canvas.draw()

    t = []
    show_steps = 2500
    batch_success = []
    success_rate = []
    sendr1_loss = []
    recvr1_loss = []
    sendr2_loss = []
    recvr2_loss = []
    success_rate_avg = []
    sender = agent1
    receiver = agent2
    for i in range(1, n_episodes + 1):
        # results["episode"] = i
        results["sender_name"] = sender.get_active_name()
        results["receiver_name"] = receiver.get_active_name()
        game.reset()
        sender_state, img_ids = game.get_sender_state(n_images=n_active_images, unique_categories=True,
                                                      return_ids=True)
        results["active_images"] = ":".join([str(x) for x in img_ids])
        results["target_image"] = img_ids[0]
        sender_action, sender_action_probs = sender.act(sender_state, explore=explore,
                                                        gibbs_temperature=gibbs_temperature)
        results["chosen_symbol"] = sender_action
        results["chosen_symbol_p"] = sender_action_probs[sender_action]
        receiver_state, img_ids = game.get_receiver_state(sender_action, return_ids=True)
        receiver_action, receiver_action_probs = receiver.act(receiver_state, explore=explore,
                                                              gibbs_temperature=gibbs_temperature)
        results["chosen_image"] = img_ids[receiver_action]
        results["chosen_image_p"] = receiver_action_probs[receiver_action]
        sender_reward, receiver_reward, success = game.evaluate_guess(receiver_action)
        sender.remember(sender_state, sender_action, sender_reward)
        receiver.remember(receiver_state, receiver_action, receiver_reward)
        results["success"] = int(success)
        batch_success.append(success)
        df_results = df_results.append(results, ignore_index=True)

        if not i % batch_size:
            if roles == "switch":
                sender.switch_role()
                receiver.switch_role()
                tmp = sender
                sender = receiver
                receiver = tmp

        if not i % 200:
            avg_success = df_results[-200:]["success"].sum() / 200
            print(f"Episode {i}, average success: {avg_success}")

    df_results.to_csv(res_file, line_terminator="\n")
    print("Test finished")
    # plt.show(block=True)
