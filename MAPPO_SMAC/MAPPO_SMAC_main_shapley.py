import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import csv
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_smac_shapley import MAPPO_SMAC
from smac.env import StarCraft2Env


class Runner_MAPPO_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize the StarCraft2 environment and extract environment info
        self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]
        self.args.obs_dim = self.env_info["obs_shape"]
        self.args.state_dim = self.env_info["state_shape"]
        self.args.action_dim = self.env_info["n_actions"]
        self.args.episode_limit = self.env_info["episode_limit"]

        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Initialize the MAPPO agents and the replay buffer
        self.agent_n = MAPPO_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize TensorBoard writer for logging
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed)
        )

        self.win_rates = []
        self.eval_steps = []
        self.total_steps = 0

        # To store Shapley rewards and baseline rewards at evaluation intervals
        self.shapley_rewards = []
        self.original_rewards = []
        self.shapley_eval_steps = []

        # Initialize reward normalization or scaling if specified
        if self.args.use_reward_norm:
            print("------ Using reward normalization ------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------ Using reward scaling ------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

        # Create folder for training data if it does not exist
        os.makedirs('./data_train', exist_ok=True)

        # Set seaborn theme and initialize live plots for win rate and rewards
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Win Rate')
        self.ax.set_title(self.env_name)
        self.ax.legend(loc='lower right')
        self.fig.show()

        self.fig_shapley, self.ax_shapley = plt.subplots(figsize=(8, 6))
        self.lines_shapley = []  # One line per agent
        self.line_original = None  # Line for baseline reward
        self.ax_shapley.set_xlabel('Training Steps')
        self.ax_shapley.set_ylabel('Reward')
        self.ax_shapley.set_title(f"{self.env_name} - Shapley vs Baseline Reward")
        self.ax_shapley.legend(loc='lower right')
        self.fig_shapley.show()

    def run(self):
        evaluate_num = -1  # Evaluation counter
        shapley_rewards_temp = []  # Temporary storage for Shapley rewards
        original_rewards_temp = []  # Temporary storage for baseline rewards
        last_interval = 0  # Last step count when evaluation data was updated

        # Main training loop
        while self.total_steps < self.args.max_train_steps:
            # Evaluate policy periodically
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            # Run one episode and update total steps
            _, _, episode_steps = self.run_episode_smac(evaluate=False)
            self.total_steps += episode_steps

            # Train agent when replay buffer is full
            if self.replay_buffer.episode_num == self.args.batch_size:
                shapley_reward, true_reward = self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

                shapley_rewards_temp.append(shapley_reward)
                original_rewards_temp.append(np.mean(true_reward))  # Using average global reward

                # Update reward plots every 20k training steps
                if self.total_steps - last_interval >= 20000:
                    rewards_array = np.array(shapley_rewards_temp)
                    avg_shapley_20k = np.mean(rewards_array, axis=0)
                    self.shapley_rewards.append(avg_shapley_20k)

                    original_rewards_array = np.array(original_rewards_temp)
                    avg_original_20k = np.mean(original_rewards_array) * 0.2
                    self.original_rewards.append(avg_original_20k)

                    self.shapley_eval_steps.append(self.total_steps)
                    self.plot_shapley_rewards()
                    self.save_shapley_csv()

                    shapley_rewards_temp = []
                    original_rewards_temp = []
                    last_interval = self.total_steps

        # Final evaluation and closing procedures
        self.evaluate_policy()
        self.env.close()
        self.save_eval_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        # Execute multiple evaluation episodes
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times

        self.win_rates.append(win_rate)
        self.eval_steps.append(self.total_steps)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(
            self.total_steps, win_rate, evaluate_reward))
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)

        # Save the current model state
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        self.plot_eval_win_rates()
        self.save_eval_csv()

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Win Rate'])
            for step, win_rate in zip(self.eval_steps, self.win_rates):
                writer.writerow([step, win_rate])

    def save_shapley_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}_shapley.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Training Steps'] + [f'Agent {i+1} Shapley' for i in range(self.args.N)] + ['Baseline Reward']
            writer.writerow(header)
            for step, shapley, baseline in zip(self.shapley_eval_steps, self.shapley_rewards, self.original_rewards):
                row = [step] + list(shapley) + [baseline]
                writer.writerow(row)

    def plot_eval_win_rates(self):
        # Update win rate plot with current evaluation data
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.win_rates)
        self.ax.relim()
        self.ax.autoscale_view()

        def dynamic_formatter(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{int(x)}'
        self.ax.xaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(
            self.env_name, self.number, self.seed))

    def plot_shapley_rewards(self):
        # Initialize reward lines if they haven't been created yet
        if not self.lines_shapley:
            for agent in range(self.args.N):
                line, = self.ax_shapley.plot([], [], label=f'Agent {agent+1} Shapley')
                self.lines_shapley.append(line)
            self.line_original, = self.ax_shapley.plot([], [], label='Baseline Reward', 
                                                         color='black', linestyle='--')
            self.ax_shapley.legend(loc='lower right')

        # Update each agent's Shapley reward line
        for agent, line in enumerate(self.lines_shapley):
            rewards_agent = [reward[agent] for reward in self.shapley_rewards]
            line.set_xdata(self.shapley_eval_steps)
            line.set_ydata(rewards_agent)

        # Update baseline reward line
        self.line_original.set_xdata(self.shapley_eval_steps)
        self.line_original.set_ydata(self.original_rewards)

        self.ax_shapley.relim()
        self.ax_shapley.autoscale_view()

        def dynamic_formatter(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{int(x)}'
        self.ax_shapley.xaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

        self.fig_shapley.canvas.draw()
        self.fig_shapley.canvas.flush_events()
        self.fig_shapley.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_shapley.png'.format(
            self.env_name, self.number, self.seed))

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()

        # Reset reward scaling if needed at the beginning of each episode
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        # Reset RNN hidden states if using recurrent networks
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None

        # Execute steps within the episode
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()            # Observations for all agents
            s = self.env.get_state()                # Global state
            avail_a_n = self.env.get_avail_actions()  # Available actions per agent
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)
            v_n = self.agent_n.get_value(s, obs_n)
            r, done, info = self.env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif self.args.use_reward_scaling:
                    r = self.reward_scaling(r)
                # Determine if the episode ended early (before reaching the step limit)
                dw = True if done and episode_step + 1 != self.args.episode_limit else False
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)

            if done:
                break

        if not evaluate:
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return win_tag, episode_reward, episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluation frequency in training steps")
    parser.add_argument("--evaluate_times", type=int, default=32, help="Number of evaluation episodes")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Frequency to save the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of episodes per training batch")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size for training")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Hidden dimension for the RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Hidden dimension for the MLP")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Hidden dimension for alliance module")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--K_epochs", type=int, default=15, help="Number of epochs per training update")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient for the policy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon to 1e-5")
    parser.add_argument("--use_relu", type=bool, default=True, help="Use ReLU activation (otherwise tanh)")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Add agent ID to observations")
    parser.add_argument("--use_agent_specific", type=bool, default=True, help="Use agent-specific global state")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Use value clipping")

    args = parser.parse_args()
    env_names = ['3m', '8m', '2s3z']
    env_index = 0
    runner = Runner_MAPPO_SMAC(args, env_name=env_names[env_index], number=1, seed=0)
    runner.run()
