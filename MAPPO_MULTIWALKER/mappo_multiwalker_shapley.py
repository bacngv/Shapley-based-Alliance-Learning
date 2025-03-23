import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SequentialSampler

# -------------------------------------------------------------------------------
# Orthogonal Initialization Function
# -------------------------------------------------------------------------------
def orthogonal_init(layer, gain=1.0):
    """
    Apply orthogonal initialization to the weights of a layer and set its biases to zero.
    
    Args:
        layer (nn.Module): The neural network layer to initialize.
        gain (float): Gain factor for the orthogonal initialization.
    """
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)  # Set bias to zero
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)  # Apply orthogonal initialization to weights

# -------------------------------------------------------------------------------
# Actor & Critic Models for Continuous Actions
# -------------------------------------------------------------------------------

# ----------------------------
# RNN-based Actor Network
# ----------------------------
class Actor_RNN(nn.Module):
    """
    RNN-based actor network for continuous actions.
    Processes input via a fully connected layer, updates a GRU hidden state,
    and outputs the mean and log standard deviation for the action distribution.
    """
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None  # GRU hidden state
        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)  # Fully connected layer
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # GRU cell
        self.fc_mean = nn.Linear(args.rnn_hidden_dim, args.action_dim)  # Outputs action mean
        self.fc_log_std = nn.Linear(args.rnn_hidden_dim, args.action_dim)  # Outputs log standard deviation
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]  # Choose activation function based on args

        # Apply orthogonal initialization if specified
        if args.use_orthogonal_init:
            print("Using orthogonal initialization")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc_mean, gain=0.01)
            orthogonal_init(self.fc_log_std, gain=0.01)

    def forward(self, actor_input):
        # Pass input through the first layer and activation function
        x = self.activate_func(self.fc1(actor_input))
        # Update the GRU hidden state with the current input
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        # Compute the action mean with tanh squashing
        action_mean = torch.tanh(self.fc_mean(self.rnn_hidden))
        # Compute the log standard deviation and clamp its values
        log_std = torch.clamp(self.fc_log_std(self.rnn_hidden), min=-20, max=2)
        return action_mean, log_std

# ----------------------------
# MLP-based Actor Network
# ----------------------------
class Actor_MLP(nn.Module):
    """
    MLP-based actor network for continuous actions.
    Consists of two hidden layers, outputting the mean and log standard deviation.
    """
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)  # Second hidden layer
        self.mean_layer = nn.Linear(args.mlp_hidden_dim, args.action_dim)  # Output for action mean
        self.log_std_layer = nn.Linear(args.mlp_hidden_dim, args.action_dim)  # Output for log standard deviation
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]  # Select activation function

        # Apply orthogonal initialization if specified
        if args.use_orthogonal_init:
            print("Using orthogonal initialization")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)
            orthogonal_init(self.log_std_layer, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))  # Pass through first layer and activation
        x = self.activate_func(self.fc2(x))  # Pass through second layer and activation
        mean = self.mean_layer(x)  # Compute action mean
        # Compute log std and clamp its values
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)
        return mean, log_std

# ----------------------------
# RNN-based Critic Network
# ----------------------------
class Critic_RNN(nn.Module):
    """
    RNN-based critic network that estimates the state value.
    Processes input via a fully connected layer, updates a GRU hidden state,
    and outputs a scalar value.
    """
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None  # GRU hidden state
        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)  # Fully connected layer
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # GRU cell
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)  # Output for state value
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]  # Activation function
        
        # Apply orthogonal initialization if specified
        if args.use_orthogonal_init:
            print("Using orthogonal initialization")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))  # Process input through fc layer and activation
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)  # Update hidden state
        value = self.fc2(self.rnn_hidden)  # Compute state value
        return value

# ----------------------------
# MLP-based Critic Network
# ----------------------------
class Critic_MLP(nn.Module):
    """
    MLP-based critic network that estimates the state value.
    Uses two hidden layers and one output layer.
    """
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)  # Output layer for state value
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]  # Activation function
        
        # Apply orthogonal initialization if specified
        if args.use_orthogonal_init:
            print("Using orthogonal initialization")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))  # Process through first hidden layer
        x = self.activate_func(self.fc2(x))  # Process through second hidden layer
        value = self.fc3(x)  # Compute state value
        return value

# -------------------------------------------------------------------------------
# Embedding and Alliance Value Networks
# -------------------------------------------------------------------------------

# ----------------------------
# Embedding Network (PhiNet)
# ----------------------------
class PhiNet(nn.Module):
    """
    Embedding network that maps observations to a latent embedding space.
    Consists of two fully connected layers with ReLU activations.
    """
    def __init__(self, obs_dim, embed_dim):
        super(PhiNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, embed_dim)  # First embedding layer
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # Second embedding layer
        self.relu = nn.ReLU()  # ReLU activation
    
    def forward(self, obs):
        x = self.relu(self.fc1(obs))  # First layer transformation with activation
        embedding = self.relu(self.fc2(x))  # Second layer transformation with activation
        return embedding

# ----------------------------
# Alliance Value Network
# ----------------------------
class AllianceValueNet(nn.Module):
    """
    Network to predict the alliance (coalition) value based on embeddings.
    Uses a hidden layer and outputs a scalar value representing the alliance value.
    """
    def __init__(self, embed_dim, hidden_dim):
        super(AllianceValueNet, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer for alliance value
        self.relu = nn.ReLU()  # ReLU activation
    
    def forward(self, global_embedding):
        x = self.relu(self.fc1(global_embedding))  # Hidden layer transformation
        alliance_value = self.fc2(x)  # Compute alliance value
        return alliance_value

# -------------------------------------------------------------------------------
# Functions for Computing Shapley Values and Reward Allocation
# -------------------------------------------------------------------------------

def compute_shapley_values(obs_all, phi_net, alliance_net, num_samples=50):
    """
    Compute Shapley values for N agents using vectorized Monte Carlo sampling.
    
    Args:
        obs_all (torch.Tensor): Observations for all N agents with shape (N, obs_dim).
        phi_net (nn.Module): Network mapping observations to embeddings.
        alliance_net (nn.Module): Network predicting the alliance value.
        num_samples (int): Number of Monte Carlo samples to approximate the Shapley values.
    
    Returns:
        torch.Tensor: Shapley values for each agent (shape (N,)), clamped to non-negative values.
    """
    device = obs_all.device
    N, _ = obs_all.shape
    embed_dim = phi_net.fc2.out_features

    # Compute embeddings for all agent observations.
    embeddings = phi_net(obs_all)  # Shape: (N, embed_dim)
    # Generate random permutations for Monte Carlo sampling.
    random_vals = torch.rand(num_samples, N, device=device)
    perms = random_vals.argsort(dim=1)
    perms_embeddings = embeddings[perms]
    # Compute cumulative embeddings for each permutation.
    cum_embeddings = perms_embeddings.cumsum(dim=1)

    # Compute alliance value for an empty coalition (zero embedding).
    zero_coalition = torch.zeros(num_samples, 1, embed_dim, device=device)
    zero_val = alliance_net(zero_coalition.view(-1, embed_dim)).view(num_samples, 1)
    # Compute alliance values for cumulative embeddings.
    alliance_vals = alliance_net(cum_embeddings.view(-1, embed_dim)).view(num_samples, N)
    # Concatenate the zero coalition value to get full coalition values.
    coalition_vals = torch.cat([zero_val, alliance_vals], dim=1)
    # Compute marginal contributions as the difference between consecutive coalition values.
    marginal_contributions = coalition_vals[:, 1:] - coalition_vals[:, :-1]

    # Aggregate the marginal contributions over all permutations for each agent.
    flat_indices = perms.reshape(-1)
    flat_marginals = marginal_contributions.reshape(-1)
    shapley_values = torch.zeros(N, device=device)
    shapley_values = shapley_values.scatter_add(0, flat_indices, flat_marginals)
    shapley_values /= num_samples

    return torch.clamp(shapley_values, min=0)

def allocate_rewards(global_reward, shapley_values, alpha=0.8):
    """
    Allocate the global reward to each agent based on their Shapley value.
    
    Each agent's reward is computed as:
      r_i = alpha * (Shapley value for agent i) + (1 - alpha) * (global_reward / N)
    
    Args:
        global_reward (scalar): The total global reward.
        shapley_values (torch.Tensor): Shapley values for each agent (shape (N,)).
        alpha (float): Weighting factor between Shapley-based reward and equal division.
    
    Returns:
        torch.Tensor: The reward allocated to each agent.
    """
    N = shapley_values.shape[0]
    equal_reward = global_reward / N  # Equal share of the global reward
    allocated = alpha * shapley_values + (1 - alpha) * equal_reward
    return allocated

# -------------------------------------------------------------------------------
# New Alliance Loss Function with Order Loss based on Global Reward
# -------------------------------------------------------------------------------
def compute_alliance_loss(phi_net, alliance_net, obs_all, global_reward, num_synergy_samples=8):
    """
    Compute the combined loss for the phi_net and alliance_net with three components:
      1. Empty coalition loss: f(empty) should be 0.
      2. Order loss: f(total alliance) should match the global reward.
      3. Synergy loss: For two disjoint subsets X and Y, ensure that f(X ∪ Y) >= f(X) + f(Y).
    
    Args:
        phi_net (nn.Module): Embedding network mapping observations to embeddings.
        alliance_net (nn.Module): Network predicting alliance value from embeddings.
        obs_all (torch.Tensor): Observations for all agents with shape (N, obs_dim).
        global_reward (torch.Tensor or scalar): The total global reward for the current timestep.
        num_synergy_samples (int): Number of samples for computing synergy loss.
    
    Returns:
        torch.Tensor: The overall combined loss.
    """
    device = obs_all.device
    N = obs_all.shape[0]
    embed_dim = phi_net.fc2.out_features

    # 1. Empty coalition loss: f(empty) should be 0.
    empty_embedding = torch.zeros(1, embed_dim, device=device)
    f_empty = alliance_net(empty_embedding)
    loss_empty = (f_empty ** 2).mean()

    # 2. Order loss: f(total alliance) should match the global reward.
    total_embedding = phi_net(obs_all).sum(dim=0, keepdim=True)
    f_total = alliance_net(total_embedding)
    # Mean squared error between f_total and global_reward.
    loss_order = (f_total - global_reward).pow(2).mean()

    # 3. Synergy loss for disjoint subsets:
    embeddings = phi_net(obs_all)  # Get embeddings for all agents

    # Generate mask for subset X: each agent is included with probability 0.5.
    mask_X = (torch.rand(num_synergy_samples, N, device=device) > 0.5)
    # Ensure X is neither empty nor full.
    for i in range(num_synergy_samples):
        if mask_X[i].sum() == 0:
            mask_X[i, 0] = True
        if mask_X[i].sum() == N:
            mask_X[i, 0] = False

    # Generate mask for subset Y from the complement of X (ensuring disjointness).
    mask_Y = (torch.rand(num_synergy_samples, N, device=device) > 0.5) & (~mask_X)
    # Ensure Y is not empty; if it is, select one element from the complement of X.
    for i in range(num_synergy_samples):
        if mask_Y[i].sum() == 0:
            complement = ~mask_X[i]
            idx = complement.nonzero(as_tuple=False)[0, 0]
            mask_Y[i, idx] = True

    # Sum embeddings for subsets X, Y, and their union.
    sum_X = torch.matmul(mask_X.float(), embeddings)         # Shape: (num_synergy_samples, embed_dim)
    sum_Y = torch.matmul(mask_Y.float(), embeddings)         # Shape: (num_synergy_samples, embed_dim)
    mask_union = mask_X | mask_Y
    sum_union = torch.matmul(mask_union.float(), embeddings)   # Shape: (num_synergy_samples, embed_dim)

    # Compute alliance values for subsets X, Y, and the union.
    f_X = alliance_net(sum_X)      # Shape: (num_synergy_samples, 1)
    f_Y = alliance_net(sum_Y)      # Shape: (num_synergy_samples, 1)
    f_union = alliance_net(sum_union)  # Shape: (num_synergy_samples, 1)

    # Penalize if the sum of alliance values for X and Y exceeds that of their union.
    synergy_violation = torch.relu(f_X + f_Y - f_union)
    loss_synergy = synergy_violation.pow(2).mean()

    # Combine the three loss components with weighting factors.
    lambda_empty = 1.0
    lambda_order = 1.0
    lambda_synergy = 1.0
    loss = lambda_empty * loss_empty + lambda_order * loss_order + lambda_synergy * loss_synergy

    return loss

# -------------------------------------------------------------------------------
# MAPPO MULTIWALKER with Integrated Shapley Reward and Alliance Loss
# -------------------------------------------------------------------------------
class MAPPO_MULTIWALKER:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) implementation for the Multiwalker
    environment with integrated Shapley reward allocation and alliance loss.
    """
    def __init__(self, args):
        # Environment and training hyperparameters.
        self.N = args.N                   # Number of agents
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.max_cycles
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

        # Determine input dimensions for actor and critic networks.
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("Adding agent id to inputs")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        # Initialize actor and critic networks based on whether to use RNN.
        if self.use_rnn:
            print("Using RNN")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        # Combine actor and critic parameters for optimization.
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("Setting Adam eps")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

        # Initialize networks for Shapley reward allocation.
        self.phi_net = PhiNet(args.obs_dim, args.embed_dim)
        self.alliance_net = AllianceValueNet(args.embed_dim, args.alliance_hidden_dim)
        self.alliance_optimizer = torch.optim.Adam(
            list(self.phi_net.parameters()) + list(self.alliance_net.parameters()),
            lr=self.lr
        )

    def choose_action(self, obs_n, evaluate):
        """
        Select actions for all agents based on their observations.
        
        Args:
            obs_n (dict): Dictionary of agent observations.
            evaluate (bool): If True, choose deterministic actions; otherwise, sample from the distribution.
        
        Returns:
            tuple: Raw actions, processed actions (after tanh squashing), and log probabilities.
        """
        with torch.no_grad():
            # Sort agent keys to ensure consistent ordering.
            keys = sorted(obs_n.keys())
            obs_list = [obs_n[key] for key in keys]
            obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32)
            
            # Prepare actor inputs; optionally include agent IDs as one-hot vectors.
            actor_inputs = [obs_tensor]
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))
            actor_inputs = torch.cat(actor_inputs, dim=-1)
            
            # Forward pass through the actor network.
            mean, log_std = self.actor(actor_inputs)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
            if evaluate:
                a_n = torch.tanh(mean)  # Deterministic action using mean.
                raw_a_n = mean
                a_logprob_n = dist.log_prob(mean).sum(dim=-1)
            else:
                raw_action = dist.sample()  # Sample action stochastically.
                a_n = torch.tanh(raw_action)
                a_logprob_n = dist.log_prob(raw_action).sum(dim=-1)
                # Adjust log probabilities to account for tanh squashing.
                a_logprob_n -= torch.sum(2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action)), dim=-1)
                raw_a_n = raw_action
            
            # Convert outputs to numpy arrays and organize into dictionaries.
            a_n_np = a_n.numpy()
            raw_a_n_np = raw_a_n.numpy()
            a_logprob_np = a_logprob_n.numpy()
            
            actions_dict = {key: a_n_np[i] for i, key in enumerate(keys)}
            raw_actions_dict = {key: raw_a_n_np[i] for i, key in enumerate(keys)}
            logprob_dict = {key: a_logprob_np[i] for i, key in enumerate(keys)}
            
            return raw_actions_dict, actions_dict, logprob_dict

    def get_value(self, s):
        """
        Estimate the state value for all agents.
        
        Args:
            s (np.array): The global state.
        
        Returns:
            np.array: An array of state values for each agent.
        """
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            # Prepare critic inputs; include agent IDs if required.
            critic_inputs = [s]
            if self.add_agent_id:
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat(critic_inputs, dim=-1)
            v_n = self.critic(critic_inputs)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        """
        Train the MAPPO networks using data from the replay buffer.
        This includes reward allocation via Shapley values, GAE advantage computation,
        and optimization of both actor/critic and alliance networks.
        
        Args:
            replay_buffer: Buffer containing training data.
            total_steps (int): Total training steps completed.
        
        Returns:
            tuple: Average allocated Shapley reward and average true global reward.
        """
        batch = replay_buffer.get_training_data()
        true_rewards = batch['r_n'].clone()
        # Reallocate rewards for each timestep using Shapley values.
        for i in range(batch['r_n'].shape[0]):
            for t in range(batch['r_n'].shape[1]):
                global_reward = torch.sum(batch['r_n'][i, t])
                obs_all = batch['obs_n'][i, t]
                shapley_vals = compute_shapley_values(obs_all, self.phi_net, self.alliance_net, num_samples=50)
                allocated = allocate_rewards(global_reward, shapley_vals)
                batch['r_n'][i, t] = allocated

        # Compute advantages using Generalized Advantage Estimation (GAE)
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

        # Prepare inputs for both actor and critic networks.
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # PPO optimization loop for K epochs.
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.use_rnn:
                    # Reset RNN hidden states.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    # Process each timestep sequentially.
                    for t in range(self.episode_limit):
                        mean, log_std = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        std = torch.exp(log_std)
                        dist_now = Normal(mean, std)
                        raw_a_n_t = batch['raw_a_n'][index, t].reshape(self.mini_batch_size * self.N, -1)
                        log_prob = dist_now.log_prob(raw_a_n_t).sum(dim=-1)
                        log_prob -= torch.sum(2 * (np.log(2) - raw_a_n_t - F.softplus(-2 * raw_a_n_t)), dim=-1)
                        probs_now.append(log_prob.reshape(self.mini_batch_size, self.N))
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))
                        values_now.append(v.reshape(self.mini_batch_size, self.N))
                    a_logprob_n_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    # For non-RNN (MLP) networks.
                    mean, log_std = self.actor(actor_inputs[index])
                    std = torch.exp(log_std)
                    dist_now = Normal(mean, std)
                    a_logprob_n_now = dist_now.log_prob(batch['raw_a_n'][index]).sum(dim=-1)
                    a_logprob_n_now -= torch.sum(2 * (np.log(2) - batch['raw_a_n'][index] - F.softplus(-2 * batch['raw_a_n'][index])), dim=-1)
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                # Calculate the ratio for the PPO loss.
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2)
                entropy = dist_now.entropy().sum(dim=-1)

                # Compute critic loss; optionally use value clipping.
                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                ac_loss = actor_loss.mean() + critic_loss.mean() - self.entropy_coef * entropy.mean()

                # Gradient descent step.
                self.ac_optimizer.zero_grad()
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        # Apply learning rate decay if specified.
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        # Compute and update alliance network loss using random batch samples.
        alliance_loss_total = 0.0
        count = 0
        for i in range(batch['obs_n'].shape[0]):
            t = torch.randint(low=0, high=batch['obs_n'].shape[1], size=(1,)).item()
            obs_all = batch['obs_n'][i, t]
            # Use true_rewards for global_reward calculation.
            global_reward = torch.sum(true_rewards[i, t])
            alliance_loss_total += compute_alliance_loss(self.phi_net, self.alliance_net, obs_all, global_reward, num_synergy_samples=8)
            count += 1
        alliance_loss_total = alliance_loss_total / count

        self.alliance_optimizer.zero_grad()
        alliance_loss_total.backward()
        self.alliance_optimizer.step()

        # Report average allocated Shapley reward and average true global reward.
        avg_shapley_reward = batch['r_n'].mean(dim=(0, 1)).detach().cpu().numpy()
        avg_true_reward = true_rewards.mean(dim=(0, 1)).detach().cpu().numpy()

        return avg_shapley_reward, avg_true_reward

    def lr_decay(self, total_steps):
        """
        Decay the learning rate linearly over the training steps.
        
        Args:
            total_steps (int): The current total training steps.
        """
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        """
        Prepare inputs for the actor and critic networks from the training batch.
        
        Args:
            batch: Dictionary containing training data (observations, states, etc.).
        
        Returns:
            tuple: Processed actor inputs and critic inputs.
        """
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
        """
        Save the actor network's state.
        
        Args:
            env_name (str): Name of the environment.
            number (int): An identifier number.
            seed (int): Random seed.
            total_steps (int): Total training steps completed.
        """
        torch.save(
            self.actor.state_dict(),
            "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
                env_name, number, seed, int(total_steps / 1000)
            )
        )

    def load_model(self, env_name, number, seed, step):
        """
        Load the actor network's state.
        
        Args:
            env_name (str): Name of the environment.
            number (int): An identifier number.
            seed (int): Random seed.
            step (int): Checkpoint step.
        """
        self.actor.load_state_dict(
            torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
                env_name, number, seed, step
            ))
        )
