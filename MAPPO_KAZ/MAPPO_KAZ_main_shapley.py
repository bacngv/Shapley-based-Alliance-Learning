import os 
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use dummy video driver for headless environments

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Plot styling
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_kaz_shapley import MAPPO_KAZ  
from kaz import KAZGymWrapper 
import imageio  # To save GIFs
from IPython import display as ipy_display  # For displaying GIFs in notebooks
from matplotlib.ticker import FuncFormatter

class Runner_KAZ:
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed

        # Set random seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        sns.set_theme(style="whitegrid", font_scale=1.2)

        self.env = KAZGymWrapper(render_mode="rgb_array")
        self.args.N = len(self.env.agents)  # Number of agents
        agent0 = self.env.agents[0]
        self.args.obs_dim = int(np.prod(self.env.observation_space[agent0].shape))
        self.args.action_dim = self.env.action_space[agent0].n
        self.args.state_dim = self.args.N * self.args.obs_dim  # Global state dimension

        print("Agents =", self.env.agents)
        print("observation_space =", self.env.observation_space)
        print("obs_dim =", self.args.obs_dim)
        print("action_space =", self.env.action_space)
        print("action_dim =", self.args.action_dim)
        print("state_dim =", self.args.state_dim)

        self.agent = MAPPO_KAZ(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.writer = SummaryWriter(log_dir='runs/KAZ/KAZ_seed_{}'.format(self.seed))
        
        # Evaluation tracking
        self.evaluate_rewards = []
        self.eval_steps = []
        self.total_steps = 0

        # Shapley and baseline rewards tracking for plotting
        self.shapley_rewards = []       
        self.shapley_eval_steps = []    
        self.original_rewards = []      
        self.lines_shapley = []         
        self.line_original = None       

        os.makedirs('./data_train', exist_ok=True)

        # Set up live plot for evaluation rewards
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        (self.line,) = self.ax.plot([], [], color='orange', label='KAZ')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        self.ax.set_title('Knights Archers Zombies')
        self.ax.legend(loc='lower right')
        self.fig.show()

        # Set up live plot for Shapley rewards
        self.fig_shapley, self.ax_shapley = plt.subplots(figsize=(8,6))
        self.ax_shapley.set_xlabel('Training Steps')
        self.ax_shapley.set_ylabel('Reward')
        self.ax_shapley.set_title('Shapley vs Baseline Reward - KAZ')
        self.ax_shapley.legend(loc='lower right')
        self.fig_shapley.show()

        # Reward normalization or scaling if enabled
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)
        
        # Initialize GIF saving step
        self.next_save_step = 20000

    def run(self):
        evaluate_num = -1
        shapley_rewards_temp = []     
        original_rewards_temp = []    
        last_interval = 0         

        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            episode_reward, episode_steps = self.run_episode(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                # Train agent and obtain Shapley and true rewards
                avg_shapley_reward, true_reward = self.agent.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()
                
                shapley_rewards_temp.append(avg_shapley_reward)
                original_rewards_temp.append(np.mean(true_reward))
                
                # Update every 20K steps
                if self.total_steps - last_interval >= 20000:
                    rewards_array = np.array(shapley_rewards_temp)  
                    avg_20k = np.mean(rewards_array, axis=0)          
                    baseline_avg = np.mean(original_rewards_temp) * 0.2  
                    
                    self.shapley_eval_steps.append(self.total_steps)
                    self.shapley_rewards.append(avg_20k)
                    self.original_rewards.append(baseline_avg)
                    
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
            episode_reward, _ = self.run_episode(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward /= self.args.evaluate_times
        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_KAZ', evaluate_reward, global_step=self.total_steps)
        
        # Save the model checkpoint
        self.agent.save_model("KAZ", 1, self.seed, self.total_steps)
        
        self.save_eval_csv()
        self.plot_eval_rewards()

        # Save a GIF if the step threshold is reached
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

    def save_shapley_csv(self):
        csv_filename = './data_train/KAZ_seed_{}_shapley.csv'.format(self.seed)
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
        self.fig.savefig('./data_train/KAZ_seed_{}_eval.png'.format(self.seed))

    def plot_shapley_rewards(self):
        # Initialize plot lines for each agent and baseline reward if not already done
        if not self.lines_shapley:
            for agent in range(self.args.N):
                line, = self.ax_shapley.plot([], [], label=f'Agent {agent+1} Shapley')
                self.lines_shapley.append(line)
            if self.line_original is None:
                self.line_original, = self.ax_shapley.plot([], [], label='Baseline Reward', 
                                                            color='black', linestyle='--')
            self.ax_shapley.legend(loc='lower right')
        
        # Update Shapley reward lines for each agent
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
        self.fig_shapley.savefig('./data_train/KAZ_seed_{}_shapley.png'.format(self.seed))

    def render_and_save_gif(self, filename='KAZ.gif'):
        frames = []
        obs_dict = self.env.reset()
        done = False
        # Capture frames during an episode
        while not done:
            frame = self.env.render(mode='rgb_array')
            frames.append(frame)
            # Prepare observation array for agents
            obs_list = [obs_dict[agent].flatten() for agent in self.env.agents]
            obs_n = np.stack(obs_list, axis=0)
            a_n, _ = self.agent.choose_action(obs_n, evaluate=True)
            actions = {agent: a_n[i] for i, agent in enumerate(self.env.agents)}
            obs_dict, _, done, _ = self.env.step(actions)
        imageio.mimsave(filename, frames, fps=30)
        print("Saved GIF to", filename)
        ipy_display.display(ipy_display.Image(filename=filename))

    def run_episode(self, evaluate=False):
        episode_reward = 0
        obs_dict = self.env.reset()
        # Process initial observations
        obs_list = [obs_dict[agent].flatten() for agent in self.env.agents]
        obs_n = np.stack(obs_list, axis=0)
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent.actor.rnn_hidden = None
            self.agent.critic.rnn_hidden = None
        
        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent.choose_action(obs_n, evaluate=evaluate)
            state = np.concatenate(obs_n, axis=0)
            v_n = self.agent.get_value(state)
            
            actions = {agent: a_n[i] for i, agent in enumerate(self.env.agents)}
            next_obs_dict, r_dict, done, _ = self.env.step(actions)
            
            next_obs_list = [next_obs_dict[agent].flatten() for agent in self.env.agents]
            obs_next_n = np.stack(next_obs_list, axis=0)
            r_list = [r_dict[agent] for agent in self.env.agents]
            r_n = np.array(r_list)
            
            episode_reward += r_n.sum()
            
            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
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
    parser = argparse.ArgumentParser("Hyperparameters for MAPPO in KAZ environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum training steps")
    parser.add_argument("--episode_limit", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluation frequency (steps)")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Number of evaluations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="RNN hidden layer size")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="MLP hidden layer size")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Alliance hidden layer size")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU (otherwise tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent ID")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")

    args = parser.parse_args()
    for seed in [1, 2, 3]:
        print(f"Running training with seed {seed}")
        runner = Runner_KAZ(args, seed=seed)
        runner.run()
    #runner = Runner_KAZ(args, seed=0)
    #runner.run()
