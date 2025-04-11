import os 
os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_multiwalker_shapley import MAPPO_MULTIWALKER
from multiwalker import GymMultiWalkerWrapper
from IPython import display as ipy_display
from matplotlib.ticker import FuncFormatter
import imageio

class Runner_MAPPO_MULTIWALKER:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        sns.set_theme(style="whitegrid", font_scale=1.2)
        discrete = False

        # Initialize the multiwalker environment (3 agents, fixed terrain length)
        self.env = GymMultiWalkerWrapper(n_walkers=3, terrain_length=500, max_cycles=args.max_cycles)
        self.args.N = 3  
        self.args.obs_dim_n = [self.env.observation_space.shape[0] for _ in range(3)]
        if discrete:
            self.args.action_dim_n = [self.env.action_space.n for _ in range(3)]
        else:
            self.args.action_dim_n = [self.env.action_space.shape[0] for _ in range(3)]
        
        # Set state and action dimensions based on the first agent (assumed identical)
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = self.args.obs_dim * 3

        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Initialize the MAPPO agent and replay buffer
        self.agent_n = MAPPO_MULTIWALKER(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize Tensorboard writer for logging purposes
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Data structures for storing evaluation and reward metrics
        self.evaluate_rewards = []
        self.eval_steps = []
        self.total_steps = 0

        self.shapley_rewards = []       # Shapley value rewards for each agent
        self.original_rewards = []      # Baseline reward (global_reward / N)
        self.shapley_eval_steps = []    # Training steps corresponding to shapley rewards

        self.next_save_step = 20000
        os.makedirs('./data_train', exist_ok=True)

        # Setup live plot for evaluation rewards
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        env_title = self.env_name.replace("simple", "").replace("_", "").strip().capitalize()
        self.ax.set_title(env_title)
        self.ax.legend(loc='lower right')
        self.fig.show()

        # Setup live plot for Shapley and baseline rewards
        self.fig_shapley, self.ax_shapley = plt.subplots(figsize=(8, 6))
        self.lines_shapley = []
        self.line_original = None  # For baseline reward
        self.ax_shapley.set_xlabel('Training Steps')
        self.ax_shapley.set_ylabel('Reward')
        self.ax_shapley.set_title(f"{env_title} - Shapley vs Baseline Reward")
        self.fig_shapley.show()

        # Configure reward normalization or scaling if enabled
        if self.args.use_reward_norm:
            print("------use reward normalization------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1
        shapley_rewards_temp = []
        original_rewards_temp = []  # Temporary storage for baseline rewards per training interval
        last_interval = 0

        while self.total_steps < self.args.max_train_steps:
            # Periodically evaluate the policy
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            # Run a single episode and accumulate its steps
            _, episode_steps = self.run_episode(evaluate=False)
            self.total_steps += episode_steps

            # Perform training when a batch of episodes is collected
            if self.replay_buffer.episode_num == self.args.batch_size:
                avg_shapley_reward, avg_true_reward = self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

                shapley_rewards_temp.append(avg_shapley_reward)
                original_rewards_temp.append(np.mean(avg_true_reward))

                # Log aggregated rewards every 20,000 training steps
                if self.total_steps - last_interval >= 20000:
                    shapley_rewards_array = np.array(shapley_rewards_temp)
                    avg_shapley_20k = np.mean(shapley_rewards_array, axis=0)
                    self.shapley_rewards.append(avg_shapley_20k)

                    original_rewards_array = np.array(original_rewards_temp)
                    avg_original_20k = np.mean(original_rewards_array) * 0.2  # New baseline calculation
                    self.original_rewards.append(avg_original_20k)

                    self.shapley_eval_steps.append(self.total_steps)
                    self.plot_shapley_rewards()
                    self.save_shapley_csv()

                    shapley_rewards_temp = []
                    original_rewards_temp = []
                    last_interval = self.total_steps

        # Final evaluation and cleanup after training is complete
        self.evaluate_policy()
        self.env.close()
        self.save_eval_csv()
        self.save_shapley_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward /= self.args.evaluate_times

        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        # Save model and update evaluation plots
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)
        self.save_eval_csv()
        self.plot_eval_rewards()

        # Save a GIF of the agent's performance periodically
        if self.total_steps >= self.next_save_step:
            gif_filename = './data_train/{}_steps_{}.gif'.format(self.env_name, self.total_steps)
            self.render_and_save_gif(gif_filename)
            self.next_save_step += 20000

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

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

    def plot_eval_rewards(self):
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
        self.ax.relim()
        self.ax.autoscale_view()

        # Format x-axis values to display in K or M as needed
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
        # Initialize plot lines for each agent if not already created
        if not self.lines_shapley:
            for agent in range(self.args.N):
                line, = self.ax_shapley.plot([], [], label=f'Agent {agent+1} Shapley')
                self.lines_shapley.append(line)
            self.line_original, = self.ax_shapley.plot([], [], label='Baseline Reward', 
                                                      color='black', linestyle='--')
            self.ax_shapley.legend(loc='lower right')

        # Update each agent's reward line
        for agent, line in enumerate(self.lines_shapley):
            rewards_agent = [reward[agent] for reward in self.shapley_rewards]
            line.set_xdata(self.shapley_eval_steps)
            line.set_ydata(rewards_agent)

        # Update the baseline reward line
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

    def run_episode(self, evaluate=False):
        episode_reward = 0
        obs, _ = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None

        for episode_step in range(self.args.max_cycles):
            # Get actions and log probabilities from the agent
            raw_a_n, a_n, a_logprob_n = self.agent_n.choose_action(obs, evaluate=evaluate)
            a_n_array = np.stack([a_n[key] for key in sorted(a_n.keys())])
            raw_a_n_array = np.stack([raw_a_n[key] for key in sorted(raw_a_n.keys())])
            a_logprob_array = np.stack([a_logprob_n[key] for key in sorted(a_logprob_n.keys())])
            
            # Concatenate observations from all agents to form the global state
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
        ipy_display.display(ipy_display.Image(filename=filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in Multiwalker Environment")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help="Maximum training steps")
    parser.add_argument("--max_cycles", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate policy every evaluate_freq steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Number of evaluations")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of episodes per training batch")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Number of episodes per mini-batch")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Number of neurons in RNN hidden layer")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Number of neurons in MLP hidden layer")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Number of neurons in alliance network hidden layer")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for policy")
    parser.add_argument("--K_epochs", type=int, default=15, help="Number of update epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient for policy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU instead of tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Add agent id information to input")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Use value clipping")

    args = parser.parse_args()
    runner = Runner_MAPPO_MULTIWALKER(args, env_name="multiwalker", number=1, seed=0)
    runner.run()
