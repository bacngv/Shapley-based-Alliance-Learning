import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import numpy as np
import copy

def orthogonal_init(layer, gain=1.0):
    """Apply orthogonal initialization to weights and zero-initialize biases."""
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)
        # Mask unavailable actions
        x[avail_a_n == 0] = -1e10
        prob = torch.softmax(x, dim=-1)
        return prob

class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value

class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        # Mask unavailable actions
        x[avail_a_n == 0] = -1e10
        prob = torch.softmax(x, dim=-1)
        return prob

class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value

# Subnetworks for Shapley Value computation
class PhiNet(nn.Module):
    def __init__(self, obs_dim, embed_dim):
        super(PhiNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        embedding = self.relu(self.fc2(x))
        return embedding

class AllianceValueNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(AllianceValueNet, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, global_embedding):
        x = self.relu(self.fc1(global_embedding))
        alliance_value = self.fc2(x)
        return alliance_value

def compute_shapley_values(obs_all, phi_net, alliance_net, num_samples=50):
    """
    Compute approximate Shapley values for agents using Monte Carlo sampling.

    Args:
        obs_all (torch.Tensor): Tensor of shape (N, obs_dim) with agents' observations.
        phi_net (nn.Module): Network that maps observations to embeddings.
        alliance_net (nn.Module): Network that predicts coalition value from an embedding.
        num_samples (int): Number of Monte Carlo samples.

    Returns:
        torch.Tensor: Non-negative Shapley values (shape: (N,)).
    """
    device = obs_all.device
    N, _ = obs_all.shape
    embed_dim = phi_net.fc2.out_features

    embeddings = phi_net(obs_all)  # (N, embed_dim)
    random_vals = torch.rand(num_samples, N, device=device)
    perms = random_vals.argsort(dim=1)
    perms_embeddings = embeddings[perms]
    cum_embeddings = perms_embeddings.cumsum(dim=1)

    zero_coalition = torch.zeros(num_samples, 1, embed_dim, device=device)
    zero_val = alliance_net(zero_coalition.view(-1, embed_dim)).view(num_samples, 1)
    alliance_vals = alliance_net(cum_embeddings.view(-1, embed_dim)).view(num_samples, N)
    coalition_vals = torch.cat([zero_val, alliance_vals], dim=1)

    marginal_contributions = coalition_vals[:, 1:] - coalition_vals[:, :-1]
    flat_indices = perms.reshape(-1)
    flat_marginals = marginal_contributions.reshape(-1)
    shapley_values = torch.zeros(N, device=device)
    shapley_values = shapley_values.scatter_add(0, flat_indices, flat_marginals)
    shapley_values /= num_samples

    return torch.clamp(shapley_values, min=0)

def allocate_rewards(global_reward, shapley_values):
    """
    Allocate global reward proportionally to agents' Shapley values.

    Args:
        global_reward (torch.Tensor): Scalar global reward.
        shapley_values (torch.Tensor): Tensor of shape (N,).

    Returns:
        torch.Tensor: Allocated rewards for each agent (shape: (N,)).
    """
    total_shapley = torch.sum(shapley_values)
    if total_shapley.item() > 0:
        allocation = global_reward * (shapley_values / total_shapley)
    else:
        allocation = torch.ones_like(shapley_values) * (global_reward / shapley_values.shape[0])
    return allocation

def compute_alliance_loss(phi_net, alliance_net, obs_all, num_permutations=5):
    """
    Compute the loss for phi_net and alliance_net with three components:
      - f(empty) should be 0.
      - The sum of marginal contributions (ordering) should equal f(total).
      - f(X) + f(Y) ≈ f(X ∪ Y) for disjoint coalitions X and Y.

    Args:
        phi_net (nn.Module): Observation-to-embedding network.
        alliance_net (nn.Module): Coalition value network.
        obs_all (torch.Tensor): Observations of all agents (shape: (N, obs_dim)).
        num_permutations (int): Number of permutations to average the ordering loss.

    Returns:
        torch.Tensor: Total alliance loss.
    """
    device = obs_all.device
    N = obs_all.shape[0]
    embed_dim = phi_net.fc2.out_features

    # Loss for empty coalition: f(empty)=0
    empty_embedding = torch.zeros(1, embed_dim, device=device)
    f_empty = alliance_net(empty_embedding)
    loss_empty = (f_empty ** 2).mean()
    
    # Ordering loss: f(total) should equal the cumulative coalition value over any permutation
    total_embedding = phi_net(obs_all).sum(dim=0, keepdim=True)
    f_total = alliance_net(total_embedding)
    loss_order = 0.0
    for _ in range(num_permutations):
        perm = torch.randperm(N, device=device)
        cumulative = torch.zeros(1, embed_dim, device=device)
        f_values = []
        for idx in perm:
            emb = phi_net(obs_all[idx].unsqueeze(0))
            cumulative += emb
            f_val = alliance_net(cumulative)
            f_values.append(f_val)
        f_perm_total = f_values[-1]
        loss_order += (f_total - f_perm_total).pow(2).mean()
    loss_order /= num_permutations
    
    # Additivity loss: f(X)+f(Y) should be close to f(X ∪ Y)
    perm = torch.randperm(N, device=device)
    split = N // 2
    X_indices = perm[:split]
    Y_indices = perm[split:]
    f_X = alliance_net(phi_net(obs_all[X_indices]).sum(dim=0, keepdim=True))
    f_Y = alliance_net(phi_net(obs_all[Y_indices]).sum(dim=0, keepdim=True))
    f_XY = alliance_net(phi_net(obs_all).sum(dim=0, keepdim=True))
    loss_add = (f_X + f_Y - f_XY).pow(2).mean()
    
    lambda_empty = 1.0
    lambda_order = 1.0
    lambda_add = 1.0
    loss = lambda_empty * loss_empty + lambda_order * loss_order + lambda_add * loss_add
    return loss

class MAPPO_SMAC:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific
        self.use_value_clip = args.use_value_clip

        # Determine input dimensions for actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        if self.use_agent_specific:
            self.critic_input_dim += args.obs_dim

        # Initialize actor and critic networks
        if self.use_rnn:
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

        # Initialize networks for Shapley Value computation
        self.phi_net = PhiNet(args.obs_dim, args.embed_dim)
        self.alliance_net = AllianceValueNet(args.embed_dim, args.alliance_hidden_dim)
        self.alliance_optimizer = torch.optim.Adam(
            list(self.phi_net.parameters()) + list(self.alliance_net.parameters()), lr=self.lr
        )

    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # (N, obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                # Append one-hot encoding for agent IDs
                actor_inputs.append(torch.eye(self.N))
            actor_inputs = torch.cat(actor_inputs, dim=-1)  # (N, actor_input_dim)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)
            prob = self.actor(actor_inputs, avail_a_n)
            if evaluate:
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s, obs_n):
        with torch.no_grad():
            critic_inputs = []
            # Repeat the global state for each agent
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            critic_inputs.append(s)
            if self.use_agent_specific:
                critic_inputs.append(torch.tensor(obs_n, dtype=torch.float32))
            if self.add_agent_id:
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat(critic_inputs, dim=-1)
            v_n = self.critic(critic_inputs)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len

        # Save original rewards before modification
        true_rewards = batch['r'].clone()

        # Integrate Shapley rewards into the batch
        for i in range(batch['r'].shape[0]):
            for t in range(batch['r'].shape[1]):
                global_reward = torch.sum(batch['r'][i, t])
                obs_all = batch['obs_n'][i, t]
                shapley_vals = compute_shapley_values(obs_all, self.phi_net, self.alliance_net, num_samples=50)
                allocated = allocate_rewards(global_reward, shapley_vals)
                batch['r'][i, t] = allocated

        # Compute advantages using Generalized Advantage Estimation (GAE)
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :-1]
            if self.use_adv_norm:
                adv_copy = copy.deepcopy(adv.numpy())
                adv_copy[batch['active'].numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))

        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)

        # Update policy over K epochs
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.use_rnn:
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(max_episode_len):
                        prob = self.actor(
                            actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1),
                            batch['avail_a_n'][index, t].reshape(self.mini_batch_size * self.N, -1)
                        )
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        values_now.append(v.reshape(self.mini_batch_size, self.N))
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index], batch['avail_a_n'][index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

        # Update phi_net and alliance_net using alliance loss
        alliance_loss_total = 0.0
        count = 0
        for i in range(batch['obs_n'].shape[0]):
            t = torch.randint(low=0, high=batch['obs_n'].shape[1], size=(1,)).item()
            obs_all = batch['obs_n'][i, t]
            alliance_loss_total += compute_alliance_loss(self.phi_net, self.alliance_net, obs_all, num_permutations=5)
            count += 1
        alliance_loss_total /= count

        self.alliance_optimizer.zero_grad()
        alliance_loss_total.backward()
        self.alliance_optimizer.step()

        avg_shapley_reward = batch['r'].mean(dim=(0,1)).detach().cpu().numpy()
        avg_true_reward = true_rewards.mean(dim=(0,1)).detach().cpu().numpy()

        return avg_shapley_reward, avg_true_reward

    def lr_decay(self, total_steps):
        """Linearly decay the learning rate."""
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.use_agent_specific:
            critic_inputs.append(batch['obs_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)
        actor_inputs = torch.cat(actor_inputs, dim=-1)
        critic_inputs = torch.cat(critic_inputs, dim=-1)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(
            self.actor.state_dict(),
            "./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(
                env_name, number, seed, int(total_steps / 1000)
            )
        )

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(
            torch.load("./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(
                env_name, number, seed, step
            ))
        )
