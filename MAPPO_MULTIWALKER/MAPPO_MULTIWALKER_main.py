import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Sử dụng dummy video driver cho SDL/pygame khi chạy trên môi trường headless (ví dụ: Colab)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Dùng seaborn để tạo style cho biểu đồ
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_multiwalker import MAPPO_MULTIWALKER
# Sử dụng Gym wrapper hỗ trợ nhiều agent cho multiwalker (parallel_env)
from multiwalker import GymMultiWalkerWrapper
from IPython import display as ipy_display  # Dùng cho live plot trên Colab nếu cần
from matplotlib.ticker import FuncFormatter  # Format nhãn trục X
import imageio  # Dùng để lưu GIF

class Runner_MAPPO_MULTIWALKER:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ---------------------------
        # Thiết lập style cho seaborn
        # ---------------------------
        sns.set_theme(style="whitegrid", font_scale=1.2)

        discrete = False
        # Tạo môi trường multiwalker với 3 agent thông qua wrapper mới
        self.env = GymMultiWalkerWrapper(n_walkers=3, terrain_length=200, max_cycles=500)
        # Cập nhật số lượng agent thành 3
        self.args.N = 3  
        # Giả sử không gian quan sát của các agent giống nhau
        self.args.obs_dim_n = [self.env.observation_space.shape[0] for _ in range(3)]
        if discrete:
            self.args.action_dim_n = [self.env.action_space.n for _ in range(3)]
        else:
            self.args.action_dim_n = [self.env.action_space.shape[0] for _ in range(3)]

        # Với các module hiện có, lấy không gian của agent đầu tiên làm tham chiếu
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        # Nếu cần trạng thái toàn cục, ghép các quan sát lại với nhau
        self.args.state_dim = self.args.obs_dim * 3

        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Tạo agent MAPPO cho đa tác tử (MAPPO_MULTIWALKER)
        self.agent_n = MAPPO_MULTIWALKER(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Khởi tạo tensorboard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Các danh sách lưu lại reward và bước evaluate
        self.evaluate_rewards = []  # Danh sách reward từ evaluate
        self.eval_steps = []        # Danh sách training steps tương ứng
        self.total_steps = 0

        # Tạo folder lưu dữ liệu nếu chưa có
        os.makedirs('./data_train', exist_ok=True)

        # Thiết lập live plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        env_title = self.env_name.replace("simple", "").replace("_", "").strip().capitalize()
        self.ax.set_title(env_title)
        self.ax.legend(loc='lower right')
        self.fig.show()

        if self.args.use_reward_norm:
            print("------use reward normalization------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1  # Số lần đánh giá đã thực hiện
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Thực hiện đánh giá policy mỗi 5k bước
                evaluate_num += 1

            _, episode_steps = self.run_episode(evaluate=False)  # Chạy một episode training
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
            episode_reward, _ = self.run_episode(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times

        # Ghi nhận bước evaluate và reward
        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        # Lưu model nếu cần
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        # Lưu CSV và cập nhật live plot
        self.save_eval_csv()
        self.plot_eval_rewards()

        # Sau mỗi 20k bước training, render và lưu GIF
        if self.total_steps % 20000 == 0:
            gif_filename = './data_train/{}_steps_{}.gif'.format(self.env_name, self.total_steps)
            self.render_and_save_gif(gif_filename)

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

    def plot_eval_rewards(self):
        # Cập nhật dữ liệu cho plot
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
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

        # Vẽ lại canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Lưu file ảnh biểu đồ
        plt.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(
            self.env_name, self.number, self.seed))

    def run_episode(self, evaluate=False):
        episode_reward = 0
        # Reset môi trường; obs là dict chứa các quan sát của 3 agent
        obs, _ = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            # Nếu dùng RNN, reset hidden state cho từng agent
            for agent in self.agent_n.agents:
                agent.actor.rnn_hidden = None
                agent.critic.rnn_hidden = None

        for episode_step in range(self.args.episode_limit):
            # Lấy cả raw_action, a_n và a_logprob_n từ policy
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
        Render environment và lưu các frame thành file GIF (chỉ dùng cho chế độ evaluate).
        """
        frames = []
        obs, _ = self.env.reset()
        done = False
        while not done:
            # Render frame dưới dạng rgb_array
            frame = self.env.render(mode='rgb_array')
            frames.append(frame)
            
            # Sử dụng policy đã train để chọn hành động
            _, a_n, _ = self.agent_n.choose_action(obs, evaluate=True)
            obs, rewards, done_flags, infos = self.env.step(a_n)
            done = any(done_flags.values())
        
        imageio.mimsave(filename, frames, fps=30)
        print("Saved GIF to", filename)
        ipy_display.display(ipy_display.Image(filename=filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in Multiwalker Environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Số bước training tối đa")
    parser.add_argument("--episode_limit", type=int, default=25, help="Số bước tối đa trong mỗi episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Đánh giá policy sau mỗi 'evaluate_freq' bước")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Số lần đánh giá")
    parser.add_argument("--batch_size", type=int, default=32, help="Số episode cho 1 batch training")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Số episode cho 1 mini batch")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Số neuron ở lớp ẩn RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Số neuron ở lớp ẩn MLP")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Số neuron ở lớp ẩn của mạng liên minh")
    parser.add_argument("--embed_dim", type=int, default=64, help="Kích thước embedding")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Hệ số chiết khấu")
    parser.add_argument("--lamda", type=float, default=0.95, help="Tham số GAE")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Tham số clipping của policy")
    parser.add_argument("--K_epochs", type=int, default=15, help="Số epoch cập nhật mỗi lần")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Có sử dụng advantage normalization hay không")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Có sử dụng reward normalization hay không")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Có sử dụng reward scaling hay không")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Hệ số entropy cho policy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Có sử dụng lr decay hay không")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Có sử dụng gradient clipping hay không")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Có sử dụng orthogonal initialization hay không")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Có set Adam epsilon=1e-5 hay không")
    parser.add_argument("--use_relu", type=bool, default=False, help="Có sử dụng ReLU hay không (nếu không, dùng tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Có sử dụng RNN hay không")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Có thêm thông tin agent id vào đầu vào hay không")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Có sử dụng value clipping hay không")

    args = parser.parse_args()
    runner = Runner_MAPPO_MULTIWALKER(args, env_name="multiwalker", number=1, seed=0)
    runner.run()
