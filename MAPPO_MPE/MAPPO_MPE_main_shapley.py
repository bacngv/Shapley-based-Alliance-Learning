import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe_shapley import MAPPO_MPE
from make_env import make_env
import os
from IPython import display as ipy_display
from matplotlib.ticker import FuncFormatter

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set seaborn style for plots
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # Create environment and set related dimensions
        discrete = True
        self.env = make_env(env_name, discrete=discrete)
        self.args.N = self.env.n  # Number of agents
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]
        if discrete:
            self.args.action_dim_n = [self.env.action_space[i].n for i in range(self.args.N)]
        else:
            self.args.action_dim_n = [self.env.action_space[i].shape[0] for i in range(self.args.N)]

        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Initialize agents and replay buffer
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Evaluation and training progress storage
        self.evaluate_rewards = []  
        self.eval_steps = []        
        self.total_steps = 0

        # Storage for Shapley rewards and baseline reward
        self.shapley_rewards = []      
        self.shapley_eval_steps = []   
        self.original_rewards = []  # Baseline reward multiplied by 0.2

        # Create folder for training data
        os.makedirs('./data_train', exist_ok=True)

        # Set up live plot for evaluation reward
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        env_title = self.env_name.replace("simple", "").replace("_", "").strip().capitalize()
        self.ax.set_title(env_title)
        self.ax.legend(loc='lower right')
        self.fig.show()

        # Set up live plot for Shapley rewards and baseline reward
        self.fig_shapley, self.ax_shapley = plt.subplots(figsize=(8, 6))
        self.lines_shapley = []  # One line per agent
        self.line_original = None  # Line for baseline reward
        self.ax_shapley.set_xlabel('Training Steps')
        self.ax_shapley.set_ylabel('Reward')
        self.ax_shapley.set_title(env_title + " - Shapley vs Baseline Reward")
        self.ax_shapley.legend(loc='lower right')
        self.fig_shapley.show()

        # Initialize reward normalization or scaling if enabled
        if self.args.use_reward_norm:
            print("Using reward normalization")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("Using reward scaling")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1  # Number of evaluations performed
        shapley_rewards_temp = []  # Temporary storage for Shapley rewards per interval
        original_rewards_temp = [] # Temporary storage for baseline rewards per interval
        last_interval = 0         # Training step count at the last update

        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                # The train function returns (average shapley reward, true reward)
                avg_shapley_reward, true_reward = self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

                shapley_rewards_temp.append(avg_shapley_reward)
                # Multiply baseline reward by 0.2 as in the SMAC version
                original_rewards_temp.append(np.mean(true_reward) * 0.2)

                if self.total_steps - last_interval >= 20000:
                    # Compute interval averages for each agent and baseline reward
                    rewards_array = np.array(shapley_rewards_temp)
                    avg_shapley_20k = np.mean(rewards_array, axis=0)
                    self.shapley_rewards.append(avg_shapley_20k)

                    avg_original_20k = np.mean(original_rewards_temp)
                    self.original_rewards.append(avg_original_20k)

                    self.shapley_eval_steps.append(self.total_steps)
                    self.plot_shapley_rewards()
                    self.save_shapley_csv()

                    shapley_rewards_temp = []
                    original_rewards_temp = []
                    last_interval = self.total_steps

        self.evaluate_policy()  # Final evaluation
        self.env.close()
        self.save_eval_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(int(self.args.evaluate_times)):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times

        # Save evaluation data
        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        # Save model if needed
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        # Update CSV file and live plot for evaluation rewards
        self.save_eval_csv()
        self.plot_eval_rewards()

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

    def plot_eval_rewards(self):
        # Update data for the evaluation reward plot
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
        self.ax.relim()
        self.ax.autoscale_view()

        # Dynamic x-axis formatter (K for thousands and M for millions)
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
        # Initialize plot lines for each agent and baseline reward, if not already created
        if not self.lines_shapley:
            for agent in range(self.args.N):
                line, = self.ax_shapley.plot([], [], label=f'Agent {agent+1} Shapley')
                self.lines_shapley.append(line)
            if self.line_original is None:
                self.line_original, = self.ax_shapley.plot([], [], label='Baseline Reward', 
                                                             color='black', linestyle='--')
            self.ax_shapley.legend(loc='lower right')

        # Update Shapley rewards for each agent
        for agent, line in enumerate(self.lines_shapley):
            rewards_agent = [reward[agent] for reward in self.shapley_rewards]
            line.set_xdata(self.shapley_eval_steps)
            line.set_ydata(rewards_agent)

        # Update baseline reward data
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

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            s = np.array(obs_n).flatten()  # Global state: concatenated observations
            v_n = self.agent_n.get_value(s)
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            episode_reward += r_n[0]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Policy evaluation frequency (steps)")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Number of evaluations per cycle")
    parser.add_argument("--batch_size", type=int, default=32, help="Episodes per training batch")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Episodes per mini-batch")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Number of neurons in RNN hidden layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Number of neurons in MLP hidden layers")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Number of neurons in alliance hidden layers")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--K_epochs", type=int, default=15, help="Training epochs per iteration")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon to 1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU activation (if False, use tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent id to observations")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=1)
    runner.run()
