import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils.plot import plot_colourline


def equivalent_error_rate(q1, err, q2, n=500):
    cdf = stats.binom.cdf(err * n, n, q1)
    return stats.binom.ppf(cdf, n, q2) / n


def run_training(game, agent1, agent2, n_episodes, batch_size, batch_mode, n_active_images,
                 roles="switch", show_plot=True, explore="gibbs", gibbs_temperature=0.01,
                 memory_sampling_distribution="linear", shared_experience=False, stop_when_goal_is_passed=False,
                 **kwargs):
    result_dict = {"goal1_reached": None, "goal2_reached": None}
    if show_plot:
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(hspace=0.65)
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        ax5 = ax4.twinx()
        plt.ion()
        fig.show()
        fig.canvas.draw()

    t = []
    analyse_batches = 200
    window = 20
    batch_success = []
    success_rate = []
    sendr1_loss = []
    recvr1_loss = []
    sendr2_loss = []
    recvr2_loss = []
    success_rate_avg = []
    success_rate_variance = []
    sender_symbols = []
    symbol_probabilities = []
    sender = agent1
    receiver = agent2

    # Goal: with 2 images (50% chance) reach 90% accuracy
    # For other number of images compute equivalent accuracy level
    goal1 = 1 - equivalent_error_rate(0.5, 0.1, 1 - 1 / n_active_images)
    goal2 = 1 - equivalent_error_rate(0.5, 0.05, 1 - 1 / n_active_images)
    goal1_reached = False
    goal2_reached = False
    print(f"Goal 1: {goal1:.4f} success rate")
    print(f"Goal 2: {goal2:.4f} success rate")

    for episode in range(1, n_episodes + 1):
        game.reset()
        sender_state = game.get_sender_state(n_images=n_active_images, unique_categories=True)
        sender_action, sender_probs = sender.act(sender_state, explore=explore, gibbs_temperature=gibbs_temperature)
        sender_symbols.append(int(sender_action))
        receiver_state = game.get_receiver_state(sender_action)
        receiver_action, _ = receiver.act(receiver_state, explore=explore, gibbs_temperature=gibbs_temperature)
        sender_reward, receiver_reward, success = game.evaluate_guess(receiver_action)

        sender.remember(sender_state, sender_action, sender_reward)
        receiver.remember(receiver_state, receiver_action, receiver_reward)

        if shared_experience:
            sender.remember(receiver_state, receiver_action, receiver_reward, net="receiver")
            receiver.remember(sender_state, sender_action, sender_reward, net="sender")

        batch_success.append(success)
        symbol_probabilities.append(sender_probs)

        if not episode % batch_size:
            avg_success = sum(batch_success) / len(batch_success)
            batch_success = []
            sender.prepare_batch(batch_size, batch_mode=batch_mode,
                                 memory_sampling_distribution=memory_sampling_distribution)
            sender.batch_train()
            receiver.prepare_batch(batch_size, batch_mode=batch_mode,
                                   memory_sampling_distribution=memory_sampling_distribution)
            receiver.batch_train()

            sender_symbols = sender_symbols[-analyse_batches:]

            if roles == "switch":
                sender.switch_role()
                receiver.switch_role()
                tmp = sender
                sender = receiver
                receiver = tmp

            # PLOT PROGRESS
            t.append(episode)
            success_rate.append(avg_success)
            if len(success_rate) < window:
                success_rate_avg.append(np.nan)
                success_rate_variance.append(np.nan)
            else:
                success_rate_avg.append(sum(success_rate[-window:]) / window)
                success_rate_variance.append(
                    sum([(x - success_rate_avg[-1])**2 for x in success_rate[-window:]]) / window)
            sendr1_loss.append(agent1.net["sender"].last_loss)
            sendr2_loss.append(agent2.net["sender"].last_loss)
            recvr1_loss.append(agent1.net["receiver"].last_loss)
            recvr2_loss.append(agent2.net["receiver"].last_loss)
            if not episode % 50:
                print(f"Episode {episode}")
                print(f"Batch success rate {success_rate_avg[-1]}, std {success_rate_variance[-1]**(1/2)}")
            if not show_plot:
                continue
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            ax1.set_title("Sender loss")
            ax2.set_title("Receiver loss")
            ax3.set_title("Batch success rate")
            ax4.set_title("Sender symbols histogram")
            ax1.plot(t[-analyse_batches:], sendr1_loss[-analyse_batches:], "m", label="Agent 1")
            ax1.plot(t[-analyse_batches:], sendr2_loss[-analyse_batches:], "c", label="Agent 2")
            ax1.legend(loc="lower left")
            ax2.plot(t[-analyse_batches:], recvr1_loss[-analyse_batches:], "m", label="Agent 1")
            ax2.plot(t[-analyse_batches:], recvr2_loss[-analyse_batches:], "c", label="Agent 2")
            ax2.legend(loc="lower left")
            ax3.plot(t[-analyse_batches:], success_rate[-analyse_batches:], "r.")
            plot_colourline(t[-analyse_batches:], success_rate_avg[-analyse_batches:],
                            success_rate_variance[-analyse_batches:], ax3)
            n_symbols = len(symbol_probabilities[0])
            histogram = ax4.hist(sender_symbols, range(n_symbols+1), align="mid")
            ax5.hlines(np.sum(symbol_probabilities, axis=0) / batch_size, range(n_symbols), range(1, n_symbols + 1),
                       linewidth=3, color="black")
            ax5.set_ylim([0, 1])
            fig.canvas.draw()
            fig.canvas.flush_events()

            symbol_probabilities = []

            result_dict["final_success_rate"] = success_rate_avg[-1]
            result_dict["symbol_histogram_median"] = np.median(histogram[0])

            if success_rate_avg[-1] >= goal1 and not goal1_reached:
                goal1_reached = True
                result_dict["goal1_reached"] = episode
                print(f"success rate > {goal1} reached in episode {episode}")
            if success_rate_avg[-1] >= goal2 and not goal2_reached:
                goal2_reached = True
                result_dict["goal1_reached"] = episode
                print(f"success rate > {goal2} reached in episode {episode}")
                if stop_when_goal_is_passed:
                    print("Stopping")
                    break

    print("Training finished")
    return result_dict


def run_test(game, agent1, agent2, res_file, n_episodes, batch_size, n_active_images,
             roles="switch", show_plot=False, explore=None, gibbs_temperature=0.01, **kwargs):
    # Set up the logging of results
    columns = "sender_name,receiver_name,active_images,target_image,chosen_symbol,chosen_symbol_p,\
    chosen_image,chosen_image_p,chosen_image_index,success".replace(" ", "").split(",")
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

    for episode in range(1, n_episodes+1):
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
        results["chosen_image_index"] = receiver_action
        sender_reward, receiver_reward, success = game.evaluate_guess(receiver_action)

        results["success"] = int(success)
        batch_success.append(success)
        df_results = df_results.append(results, ignore_index=True)

        if not episode % batch_size:
            if roles == "switch":
                sender.switch_role()
                receiver.switch_role()
                tmp = sender
                sender = receiver
                receiver = tmp

        if not episode % 200:
            avg_success = df_results[-200:]["success"].sum() / 200
            print(f"Episode {episode}, average success: {avg_success}")

    df_results.to_csv(res_file, line_terminator="\n")
    print("Test finished")
    # plt.show(block=True)
