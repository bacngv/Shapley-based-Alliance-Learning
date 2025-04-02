import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler

# Orthogonal initialization for a given layer
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

# Actor network with RNN
class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob

# Critic network with RNN
class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value

# Actor network with MLP
class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob

# Critic network with MLP
class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value

# -------------------------------
# Embedding and Alliance Value Networks
# -------------------------------

# Network for embedding observations
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

# Network to compute alliance value from a global embedding
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

# -------------------------------
# Functions for Owen Value and Reward Allocation
# -------------------------------
def compute_owen_values(obs_all, phi_net, alliance_net, coalition_ids, num_samples=50):
    """
    Vectorized computation of Owen values for N agents based on coalition structure,
    using two-level permutation sampling.
    
    Args:
        obs_all (torch.Tensor): Tensor with shape (N, obs_dim) for N agents.
        phi_net (nn.Module): Network mapping observations to embeddings.
        alliance_net (nn.Module): Network predicting the coalition value.
        coalition_ids (list or np.array): List/array of size N containing the coalition id for each agent.
        num_samples (int): Number of Monte Carlo samples.
        
    Returns:
        torch.Tensor: Owen value for each agent with shape (N,), clamped to be non-negative.
    """
    device = obs_all.device
    N, _ = obs_all.shape
    embed_dim = phi_net.fc2.out_features

    # Compute embeddings for all agents: (N, embed_dim)
    embeddings = phi_net(obs_all)  

    # Convert coalition_ids to a numpy array and get unique coalitions.
    coalition_ids = np.array(coalition_ids)
    unique_coalitions = np.unique(coalition_ids)
    
    # Map each agent to the index of its coalition in unique_coalitions (for vectorization)
    coalition_to_idx = {cid: i for i, cid in enumerate(unique_coalitions)}
    agent_coalition_idx = np.array([coalition_to_idx[cid] for cid in coalition_ids])  # (N,)
    agent_coalition_idx = torch.tensor(agent_coalition_idx, device=device)

    # Create num_samples copies of embeddings: (num_samples, N, embed_dim)
    embeddings_batch = embeddings.unsqueeze(0).expand(num_samples, N, embed_dim)

    # For each sample, create a "sorting key" for each agent:
    # 1. Each coalition receives a random value (level 1).
    # 2. Each agent gets an additional small random value (level 2) for intra-coalition permutation.
    coalition_rand = torch.rand(num_samples, len(unique_coalitions), device=device)  # (num_samples, num_coalitions)
    agent_coalition_rand = coalition_rand[:, agent_coalition_idx]  # (num_samples, N)
    # Secondary value for intra-coalition permutation
    tie_breaker = torch.rand(num_samples, N, device=device)  # (num_samples, N)
    # Combine keys: multiply coalition value by 10 (or a sufficiently large number) then add tie_breaker to maintain order
    sort_keys = agent_coalition_rand * 10 + tie_breaker  # (num_samples, N)

    # Get the sorted order for each sample (in increasing order of sort_keys)
    _, ordering = torch.sort(sort_keys, dim=1)  # ordering: (num_samples, N) containing agent indices

    # Reorder embeddings according to ordering for each sample:
    # Expand dims appropriately to gather using indices:
    ordering_expanded = ordering.unsqueeze(-1).expand(-1, -1, embed_dim)  # (num_samples, N, embed_dim)
    perm_embeddings = torch.gather(embeddings_batch, dim=1, index=ordering_expanded)  # (num_samples, N, embed_dim)

    # Compute cumulative sum along dimension 1: (num_samples, N, embed_dim)
    cum_embeddings = perm_embeddings.cumsum(dim=1)

    # Compute coalition values for empty coalition and for accumulated embeddings.
    zero_val = alliance_net(torch.zeros(1, embed_dim, device=device))  # (1, 1)
    # Expand zero_val for batch: (num_samples, 1, 1)
    zero_val_batch = zero_val.expand(num_samples, 1, 1)
    # Compute values for the accumulated embeddings: (num_samples, N, 1)
    alliance_vals = alliance_net(cum_embeddings)
    # Concatenate zero_val at the beginning for each sample: (num_samples, N+1, 1)
    coalition_vals = torch.cat([zero_val_batch, alliance_vals], dim=1)
    # Compute marginal contributions: the difference between consecutive coalition values: (num_samples, N, 1)
    marginal_contributions = coalition_vals[:, 1:] - coalition_vals[:, :-1]  # (num_samples, N, 1)
    marginal_contributions = marginal_contributions.squeeze(-1)  # (num_samples, N)

    # "Scatter" the marginal contributions back to each agent according to their original order:
    # Create a temporary tensor to hold values for each sample, initialized to zeros: (num_samples, N)
    owen_values_batch = torch.zeros(num_samples, N, device=device)
    # For each sample, assign marginal_contributions[i, j] at position ordering[i, j]
    # Using scatter_add: the source indices are marginal_contributions, and the destination indices are ordering.
    owen_values_batch = owen_values_batch.scatter_add(dim=1, index=ordering, src=marginal_contributions)

    # Average over num_samples: (N,)
    owen_values = owen_values_batch.mean(dim=0)
    
    return torch.clamp(owen_values, min=0)


def allocate_rewards_owen(global_reward, owen_values, alpha=0.8):
    """
    Allocate global_reward to each agent based on the formula:
    r_i = alpha * r_owen,i + (1 - alpha) * (r_global / N)
    
    Args:
        global_reward (scalar): The global reward.
        owen_values (torch.Tensor): Owen value for each agent (shape (N,)).
        alpha (float): Weighting factor between Owen-based reward and equal division.
    
    Returns:
        torch.Tensor: Reward allocated to each agent.
    """
    N = owen_values.shape[0]
    equal_reward = global_reward / N
    allocated = alpha * owen_values + (1 - alpha) * equal_reward
    return allocated

# -------------------------------
# Compute Alliance Loss
# -------------------------------
def compute_alliance_loss(phi_net, alliance_net, obs_all, num_permutations=5, num_synergy_samples=8):
    """
    Tính loss cho phi_net và alliance_net với 2 thành phần:
    1. f(empty) = 0.
    2. Synergy loss cho các tập hợp rời rạc X và Y: khuyến khích f(X ∪ Y) >= f(X) + f(Y).
    
    Args:
        phi_net (nn.Module): Mạng ánh xạ quan sát thành embedding.
        alliance_net (nn.Module): Mạng tính giá trị liên minh.
        obs_all (torch.Tensor): Tensor có shape (N, obs_dim) cho N tác nhân.
        num_permutations (int): Số hoán vị dùng cho order loss.
        num_synergy_samples (int): Số mẫu dùng cho synergy loss.
    
    Returns:
        torch.Tensor: Tổng loss.
    """
    device = obs_all.device
    N = obs_all.shape[0]
    embed_dim = phi_net.fc2.out_features

    # 1. Loss cho empty coalition
    empty_embedding = torch.zeros(1, embed_dim, device=device)
    f_empty = alliance_net(empty_embedding)
    loss_empty = (f_empty ** 2).mean()

    # 2. Synergy loss
    embeddings = phi_net(obs_all)  # (N, embed_dim)
    mask_X = (torch.rand(num_synergy_samples, N, device=device) > 0.5)
    for i in range(num_synergy_samples):
        if mask_X[i].sum() == 0:
            mask_X[i, 0] = True
        if mask_X[i].sum() == N:
            mask_X[i, 0] = False

    mask_Y = (torch.rand(num_synergy_samples, N, device=device) > 0.5) & (~mask_X)
    for i in range(num_synergy_samples):
        if mask_Y[i].sum() == 0:
            complement = ~mask_X[i]
            idx = complement.nonzero(as_tuple=False)[0, 0]
            mask_Y[i, idx] = True

    sum_X = torch.matmul(mask_X.float(), embeddings)
    sum_Y = torch.matmul(mask_Y.float(), embeddings)
    mask_union = mask_X | mask_Y
    sum_union = torch.matmul(mask_union.float(), embeddings)

    f_X = alliance_net(sum_X)
    f_Y = alliance_net(sum_Y)
    f_union = alliance_net(sum_union)

    synergy_violation = torch.relu(f_X + f_Y - f_union)
    loss_synergy = synergy_violation.pow(2).mean()

    lambda_empty = 1.0
    lambda_synergy = 1.0
    loss = lambda_empty * loss_empty + lambda_synergy * loss_synergy

    return loss

# MAPPO_KAZ class integrating Shapley rewards and alliance loss
class MAPPO_KAZ:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

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
        self.use_value_clip = args.use_value_clip

        # Set input dimensions for actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

        # Initialize networks for Shapley value computation
        self.phi_net = PhiNet(args.obs_dim, args.embed_dim)
        self.alliance_net = AllianceValueNet(args.embed_dim, args.alliance_hidden_dim)
        self.alliance_optimizer = torch.optim.Adam(
            list(self.phi_net.parameters()) + list(self.alliance_net.parameters()),
            lr=self.lr
        )

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))
            actor_inputs = torch.cat(actor_inputs, dim=-1)
            prob = self.actor(actor_inputs)
            if evaluate:
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            critic_inputs.append(s)
            if self.add_agent_id:
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat(critic_inputs, dim=-1)
            v_n = self.critic(critic_inputs)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()
        true_rewards = batch['r_n'].clone()

        coalition_ids = self.coalition_ids

        for i in range(batch['r_n'].shape[0]):
            for t in range(batch['r_n'].shape[1]):
                global_reward = torch.sum(batch['r_n'][i, t])
                obs_all = batch['obs_n'][i, t]
                owen_vals = compute_owen_values(obs_all, self.phi_net, self.alliance_net, coalition_ids, num_samples=50)
                allocated = allocate_rewards_owen(global_reward, owen_vals)
                batch['r_n'][i, t] = allocated

        # Compute advantages using GAE
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :-1]
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.use_rnn:
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        values_now.append(v.reshape(self.mini_batch_size, self.N))
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

        alliance_loss_total = 0.0
        count = 0
        for i in range(batch['obs_n'].shape[0]):
            t = torch.randint(low=0, high=batch['obs_n'].shape[1], size=(1,)).item()
            obs_all = batch['obs_n'][i, t]
            alliance_loss_total += compute_alliance_loss(self.phi_net, self.alliance_net, obs_all, num_permutations=5)
            count += 1
        alliance_loss_total = alliance_loss_total / count

        self.alliance_optimizer.zero_grad()
        alliance_loss_total.backward()
        self.alliance_optimizer.step()

        avg_owen_reward = batch['r_n'].mean(dim=(0, 1)).detach().cpu().numpy()
        avg_true_reward = true_rewards.mean(dim=(0, 1)).detach().cpu().numpy()

        return avg_owen_reward, avg_true_reward

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)
        actor_inputs = torch.cat(actor_inputs, dim=-1)
        critic_inputs = torch.cat(critic_inputs, dim=-1)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
            env_name, number, seed, int(total_steps / 1000)
        ))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
            env_name, number, seed, step
        )))
