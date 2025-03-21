import os 
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use dummy video driver for SDL/pygame in headless environments

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For improved plot styling
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_multiwalker_shapley import MAPPO_MULTIWALKER
from multiwalker import GymMultiWalkerWrapper  # Gym wrapper for multiwalker (parallel_env)
from IPython import display as ipy_display
from matplotlib.ticker import FuncFormatter
import imageio

class Runner_MAPPO_MULTIWALKER:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set seaborn style
        sns.set_theme(style="whitegrid", font_scale=1.2)
        discrete = False

        # Create multiwalker environment with 3 agents
        self.env = GymMultiWalkerWrapper(n_walkers=3, terrain_length=200, max_cycles=args.max_cycles)
        self.args.N = 3  
        self.args.obs_dim_n = [self.env.observation_space.shape[0] for _ in range(3)]
        if discrete:
            self.args.action_dim_n = [self.env.action_space.n for _ in range(3)]
        else:
            self.args.action_dim_n = [self.env.action_space.shape[0] for _ in range(3)]

        # Use the first agent's observation and action dimensions as reference
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = self.args.obs_dim * 3

        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Initialize MAPPO agent and replay buffer
        self.agent_n = MAPPO_MULTIWALKER(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Lists to store evaluation rewards, training steps, and Shapley values
        self.evaluate_rewards = []
        self.eval_steps = []
        self.shapley_values = []
        self.total_steps = 0

        # Set threshold for saving GIF (every 20k steps)
        self.next_save_step = 20000

        # Create folder for saving data
        os.makedirs('./data_train', exist_ok=True)

        # Setup plot for saving reward curves
        plt.ioff()  # Turn off interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        env_title = self.env_name.replace("simple", "").replace("_", "").strip().capitalize()
        self.ax.set_title(env_title)
        self.ax.legend(loc='lower right')

        if self.args.use_reward_norm:
            print("------ Using reward normalization ------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------ Using reward scaling ------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, episode_steps = self.run_episode(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()
        self.save_eval_csv()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times

        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        shapley_value = self.compute_shapley_value()
        self.shapley_values.append(shapley_value)

        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        self.save_eval_csv()
        self.plot_eval_rewards()
        self.plot_shapley_values()

        if self.total_steps >= self.next_save_step:
            gif_filename = './data_train/{}_steps_{}.gif'.format(self.env_name, self.total_steps)
            self.render_and_save_gif(gif_filename)
            self.next_save_step += 20000

    def compute_shapley_value(self):
        """
        Compute Shapley values for each agent.
        This dummy implementation returns random values.
        """
        return np.random.rand(self.args.N)

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

    def plot_eval_rewards(self):
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
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
        reward_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(
            self.env_name, self.number, self.seed)
        self.fig.savefig(reward_filename)

    def plot_shapley_values(self):
        shapley_array = np.array(self.shapley_values)  # Shape: (num evaluations, num agents)
        fig, ax = plt.subplots(figsize=(8,6))
        for agent in range(self.args.N):
            ax.plot(self.eval_steps, shapley_array[:, agent], label=f"Agent {agent+1}")
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Shapley Value')
        ax.set_title('Shapley Value Evaluation')
        ax.legend(loc='best')
        shapley_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}_shapley.png'.format(
            self.env_name, self.number, self.seed)
        fig.savefig(shapley_filename)
        plt.close(fig)

    def run_episode(self, evaluate=False):
        episode_reward = 0
        obs, _ = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None

        for episode_step in range(self.args.max_cycles):
            raw_a_n, a_n, a_logprob_n = self.agent_n.choose_action(obs, evaluate=evaluate)
            a_n_array = np.stack([a_n[key] for key in sorted(a_n.keys())])
            raw_a_n_array = np.stack([raw_a_n[key] for key in sorted(raw_a_n.keys())])
            a_logprob_array = np.stack([a_logprob_n[key] for key in sorted(a_logprob_n.keys())])
            
            s = np.concatenate([obs[agent].flatten() for agent in self.env.agents])
            v_n = self.agent_n.get_value(s)
            obs_next, r_n, done, infos = self.env.step(a_n)
            total_r = sum(r_n.values()) / len(r_n)
            episode_reward += total_r

            if not evaluate:
                r_n_array = np.array([r_n[agent] for agent in self.env.agents])
                done_n_array = np.array([done[agent] for agent in self.env.agents])
                if self.args.use_reward_norm:
                    r_n_array = self.reward_norm(r_n_array)
                elif self.args.use_reward_scaling:
                    r_n_array = self.reward_scaling(r_n_array)
                obs_array = np.stack([obs[agent] for agent in sorted(obs.keys())])
                self.replay_buffer.store_transition(
                    episode_step, obs_array, s, v_n, raw_a_n_array, a_logprob_array, r_n_array, done_n_array
                )

            obs = obs_next
            if any(done.values()):
                break

        if not evaluate:
            s = np.concatenate([obs[agent].flatten() for agent in self.env.agents])
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1

    def render_and_save_gif(self, filename='multiwalker.gif'):
        """
        Render the environment and save the frames as a GIF (for evaluation mode).
        """
        frames = []
        obs, _ = self.env.reset()
        done = False
        while not done:
            frame = self.env.render(mode='rgb_array')
            frames.append(frame)
            
            _, a_n, _ = self.agent_n.choose_action(obs, evaluate=True)
            obs, rewards, done_flags, infos = self.env.step(a_n)
            done = any(done_flags.values())
        
        imageio.mimsave(filename, frames, fps=30)
        print("Saved GIF to", filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in Multiwalker Environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum training steps")
    parser.add_argument("--max_cycles", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluation frequency in training steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Number of evaluations")
    parser.add_argument("--batch_size", type=int, default=32, help="Episodes per training batch")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Episodes per mini batch")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="RNN hidden neurons")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="MLP hidden neurons")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Alliance network hidden neurons")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for policy")
    parser.add_argument("--K_epochs", type=int, default=15, help="Update epochs per training")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient for policy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU (otherwise, tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Include agent id in input")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Use value clipping")

    args = parser.parse_args()
    runner = Runner_MAPPO_MULTIWALKER(args, env_name="multiwalker", number=1, seed=0)
    runner.run()
