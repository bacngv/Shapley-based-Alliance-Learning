import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from make_env import make_env
import os
from IPython import display as ipy_display  # Import cho live plot trên Colab

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        discrete = True
        # Create environment
        self.env = make_env(env_name, discrete=discrete)
        self.args.N = self.env.n  # Số lượng agent
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

        # Tạo N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Khởi tạo tensorboard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        # Các danh sách lưu lại reward và bước evaluate
        self.evaluate_rewards = []  # Danh sách reward từ evaluate
        self.eval_steps = []        # Danh sách training steps tương ứng
        self.total_steps = 0

        # Tạo folder lưu dữ liệu nếu chưa có
        os.makedirs('./data_train', exist_ok=True)

        # Thiết lập live plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], marker='o')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        self.ax.set_title('Spread')
        self.fig.show()

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1  # Số lần đánh giá đã thực hiện
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Thực hiện đánh giá policy
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Chạy một episode training
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Huấn luyện agent
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()  # Đánh giá lần cuối
        self.env.close()
        # Sau khi training, lưu file CSV và tắt chế độ live plot
        self.save_eval_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times

        # Ghi nhận bước evaluate và reward
        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        # Lưu numpy file (nếu cần)
        #np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
        #        np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        # Lưu CSV và cập nhật live plot
        self.save_eval_csv()
        self.plot_eval_rewards()

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])
        #print("CSV file saved at:", csv_filename)

    def plot_eval_rewards(self):
        # Cập nhật dữ liệu cho plot
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Lưu file ảnh biểu đồ
        plt.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(self.env_name, self.number, self.seed))

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
            s = np.array(obs_n).flatten()  # Global state là sự nối của tất cả các obs
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
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Number of neurons in rnn hidden layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Number of neurons in mlp hidden layers")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Number of neurons in alliance hidden layers")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
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
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU (if False, use tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent id")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=0)
    runner.run()
