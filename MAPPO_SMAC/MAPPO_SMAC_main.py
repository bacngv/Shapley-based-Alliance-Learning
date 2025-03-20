import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Dùng seaborn để tạo style
from matplotlib.ticker import FuncFormatter  # Dùng để format nhãn trục X
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import csv
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_smac import MAPPO_SMAC
from smac.env import StarCraft2Env

class Runner_MAPPO_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name  # Tên môi trường
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
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

        # Create N agents
        self.agent_n = MAPPO_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Khai báo danh sách lưu lại win rate và các bước evaluate
        self.win_rates = []
        self.eval_steps = []
        self.total_steps = 0

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

        # Tạo folder lưu dữ liệu nếu chưa có
        os.makedirs('./data_train', exist_ok=True)

        # Thiết lập style plot với seaborn tương tự mappo_mpe
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Win Rate')
        # Tiêu đề biểu đồ sẽ được đặt theo tên của môi trường
        self.ax.set_title(self.env_name)
        self.ax.legend(loc='lower right')
        self.fig.show()

    def run(self):
        evaluate_num = -1  # Số lần đánh giá đã thực hiện
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Đánh giá policy (và lưu model)
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Chạy một episode training
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Huấn luyện agent
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()  # Đánh giá lần cuối
        self.env.close()
        self.save_eval_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times

        self.win_rates.append(win_rate)
        self.eval_steps.append(self.total_steps)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)

        # Lưu win rate dưới dạng file numpy
        # np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
        #        np.array(self.win_rates))

        # Save model (giả sử agent có hàm save_model)
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        self.plot_eval_win_rates()
        self.save_eval_csv()

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Win Rate'])
            for step, win_rate in zip(self.eval_steps, self.win_rates):
                writer.writerow([step, win_rate])

    def plot_eval_win_rates(self):
        # Cập nhật dữ liệu cho plot
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.win_rates)
        self.ax.relim()
        self.ax.autoscale_view()

        # Thiết lập dynamic formatter cho trục X:
        def dynamic_formatter(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{int(x)}'
        self.ax.xaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

        # Vẽ lại canvas và lưu file ảnh biểu đồ
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(
            self.env_name, self.number, self.seed))

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # Nếu sử dụng RNN, reset hidden state trước mỗi episode
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()         # obs_n.shape=(N, obs_dim)
            s = self.env.get_state()             # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Các hành động khả dụng của N agent
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
                # Xác định trạng thái kết thúc: dw=True nếu kết thúc sớm (không phải do đạt max_episode_steps)
                dw = True if done and episode_step + 1 != self.args.episode_limit else False
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)

            if done:
                break

        if not evaluate:
            # Lưu giá trị cuối cùng của episode
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return win_tag, episode_reward, episode_step + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Hidden dimension for RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Hidden dimension for MLP")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Hidden dimension for alliance module")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=True, help="Whether to use ReLU (if False, use tanh)")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent id")
    parser.add_argument("--use_agent_specific", type=bool, default=True, help="Whether to use agent specific global state")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")

    args = parser.parse_args()
    env_names = ['3m', '8m', '2s3z']
    env_index = 0
    runner = Runner_MAPPO_SMAC(args, env_name=env_names[env_index], number=1, seed=0)
    runner.run()
