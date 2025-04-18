import os 
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use dummy video driver for headless environments (e.g., Colab)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For setting the style
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_kaz import MAPPO_KAZ  
from kaz import KAZGymWrapper 
import imageio  # Library to save GIF files
from IPython import display as ipy_display  # To display GIF in notebook
from matplotlib.ticker import FuncFormatter

class Runner_KAZ:
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed

        # Set seeds for numpy and torch
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        sns.set_theme(style="whitegrid", font_scale=1.2)

        self.env = KAZGymWrapper(render_mode="rgb_array")
        # Number of agents = length of the agents list in the environment
        self.args.N = len(self.env.agents)
        agent0 = self.env.agents[0]
        self.args.obs_dim = int(np.prod(self.env.observation_space[agent0].shape))
        self.args.action_dim = self.env.action_space[agent0].n
        # Global state is the concatenation of all agents' observations (assuming each agent has the same obs_dim)
        self.args.state_dim = self.args.N * self.args.obs_dim

        print("Agents =", self.env.agents)
        print("observation_space =", self.env.observation_space)
        print("obs_dim =", self.args.obs_dim)
        print("action_space =", self.env.action_space)
        print("action_dim =", self.args.action_dim)
        print("state_dim =", self.args.state_dim)

        self.agent = MAPPO_KAZ(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.writer = SummaryWriter(log_dir='runs/KAZ/KAZ_seed_{}'.format(self.seed))
        # Lists for storing evaluation results
        self.evaluate_rewards = []
        self.eval_steps = []
        self.total_steps = 0

        # Create folder for saving data if it does not exist
        os.makedirs('./data_train', exist_ok=True)

        # Initialize live plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        (self.line,) = self.ax.plot([], [], color='orange', label='KAZ')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        self.ax.set_title('Knights Archers Zombies')
        self.ax.legend(loc='lower right')
        self.fig.show()

        # Use reward normalization/scaling if needed
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)
        
        # Set the step at which to save a GIF (e.g., every 20000 steps)
        self.next_save_step = 20000

    def run(self):
        evaluate_num = -1  # Number of evaluations performed
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            episode_reward, episode_steps = self.run_episode(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()  # Final evaluation
        self.env.close()
        self.save_eval_csv()
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
        self.writer.add_scalar('evaluate_step_rewards_KAZ', evaluate_reward, global_step=self.total_steps)
        
        # Save the model if needed
        self.agent.save_model("KAZ", 1, self.seed, self.total_steps)
        
        self.save_eval_csv()
        self.plot_eval_rewards()

        # Save a GIF after each evaluation if the save step is reached
        if self.total_steps >= self.next_save_step:
            gif_filename = './data_train/KAZ_seed_{}_steps_{}.gif'.format(self.seed, self.total_steps)
            self.render_and_save_gif(gif_filename)
            self.next_save_step += 20000

    def save_eval_csv(self):
        csv_filename = './data_train/KAZ_seed_{}.csv'.format(self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

    def plot_eval_rewards(self):
        # Update plot data
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
        self.fig.canvas.flush_events()
        plt.savefig('./data_train/KAZ_seed_{}_eval.png'.format(self.seed))

    def render_and_save_gif(self, filename='KAZ.gif'):
        frames = []
        obs_dict = self.env.reset()
        done = False
        # Collect frames during one episode
        while not done:
            # Get frame from the environment
            frame = self.env.render(mode='rgb_array')
            frames.append(frame)
            # Convert observations from dictionary to numpy array by flattening each observation
            obs_list = []
            for agent in self.env.agents:
                obs_list.append(obs_dict[agent].flatten())
            obs_n = np.stack(obs_list, axis=0)
            # Select actions (evaluation mode: no added noise)
            a_n, _ = self.agent.choose_action(obs_n, evaluate=True)
            # Create an action dictionary for the environment
            actions = {}
            for i, agent in enumerate(self.env.agents):
                actions[agent] = a_n[i]
            obs_dict, _, done, _ = self.env.step(actions)
        # Save the collected frames as a GIF file
        imageio.mimsave(filename, frames, fps=30)
        print("Saved GIF to", filename)
        # Display the GIF if running in a notebook
        ipy_display.display(ipy_display.Image(filename=filename))

    def run_episode(self, evaluate=False):
        episode_reward = 0
        # Reset the environment and get a dictionary of agents' observations
        obs_dict = self.env.reset()
        # Convert observations: flatten each observation and order them by self.env.agents
        obs_list = []
        for agent in self.env.agents:
            obs_list.append(obs_dict[agent].flatten())
        obs_n = np.stack(obs_list, axis=0)  # shape: (N, obs_dim)
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent.actor.rnn_hidden = None
            self.agent.critic.rnn_hidden = None
        
        for episode_step in range(self.args.episode_limit):
            # Choose actions for all agents: a_n shape (N,) and a_logprob_n shape (N,)
            a_n, a_logprob_n = self.agent.choose_action(obs_n, evaluate=evaluate)
            # Create global state by concatenating all agents' observations
            state = np.concatenate(obs_n, axis=0)  # shape: (N*obs_dim,)
            v_n = self.agent.get_value(state)  # Expected to return an array of shape (N,)
            
            # Create an action dictionary for the environment
            actions = {}
            for i, agent in enumerate(self.env.agents):
                actions[agent] = a_n[i]
            
            next_obs_dict, r_dict, done, _ = self.env.step(actions)
            
            # Convert next observations: flatten each agent's observation
            next_obs_list = []
            for agent in self.env.agents:
                next_obs_list.append(next_obs_dict[agent].flatten())
            obs_next_n = np.stack(next_obs_list, axis=0)
            
            # Convert rewards: get them in the order of self.env.agents
            r_list = []
            for agent in self.env.agents:
                r_list.append(r_dict[agent])
            r_n = np.array(r_list)  # shape: (N,)
            
            # Accumulate the global reward (here using the sum of all agents' rewards; adjust as needed)
            episode_reward += r_n.sum()
            
            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                # Save the transition: storing all agents' information
                self.replay_buffer.store_transition(episode_step, obs_n, state, v_n, a_n, a_logprob_n, r_n, done)
            
            obs_n = obs_next_n
            if done:
                break

        if not evaluate:
            state = np.concatenate(obs_n, axis=0)
            v_n = self.agent.get_value(state)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in KAZ environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Number of evaluations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Number of neurons in RNN hidden layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Number of neurons in MLP hidden layers")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Number of neurons in alliance hidden layers")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU (if False, use tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent id")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")

    args = parser.parse_args()
    runner = Runner_KAZ(args, seed=0)
    runner.run()
