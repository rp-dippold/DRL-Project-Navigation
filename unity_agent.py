import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from model import DQNetwork
from collections import deque, namedtuple


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, hidden_sizes):
        """Initialize an Agent object.
        
        Parameters
        ----------
        state_size : int
            Dimension of each state
        action_size : int 
            Dimension of each action
        hidden_sizes : list
            Number of neurons in hidden layers
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.config = Config.get_config()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.dqnetwork_local = DQNetwork(
            state_size, action_size, hidden_sizes).to(self.device)
        self.dqnetwork_target = DQNetwork(
            state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.dqnetwork_local.parameters(), 
                                    lr=self.config.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, 
                                   self.config.buffer_size,
                                   self.config.batch_size,
                                   self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    

    def get_priority(self, state, action, reward, next_state, done, gamma):
        """Returns the priority for an experience tuple.

        The priority is the magnitude of the time difference (TD) error, i.e.
        p(t) = |δ(t)|.

        The TD error is defined as
            δ(t) = R(t+1) + [γ * max(a)(q^(S(t+1), a, w) - q^(S(t), A(t), w)]

        Parameters
        ----------
        state : array-like (37 elements)
            Current state (observation) of the game.
        action : int
            Action taken to reach next state
        reward : float
            Reward returned by moving from state to next_state
        next_state : array-like (37 elements)
            Next state reached by action from (current) state
        done : bool
            Indicates if the episode has finished
        gamma : float
            Discount factor

        Returns
        -------
            The p(t), i.e. the magnitude of the difference between the
            target q-value and the expected q-value
        """
        state_ts = torch.from_numpy(state).float().to(self.device)
        action_ts = torch.from_numpy(np.array(action)).long().to(self.device)
        reward_ts = torch.from_numpy(np.array(reward)).float().to(self.device)
        next_state_ts = torch.from_numpy(next_state).float().to(self.device)
        done_ts = torch.from_numpy(
            np.array(done).astype(np.uint8)).float().to(self.device)
        
        # Calculate the expected q-value q^(S(t), A(t), w)
        self.dqnetwork_local.eval()
        with torch.no_grad():
            q_expected = self.dqnetwork_local(state_ts).detach()[action_ts]
        self.dqnetwork_local.train()

        # Calculate the target q-value R(t+1) + γ * max(a)(q^(S(t+1), a, w)
        q_target_next = self.dqnetwork_target(next_state_ts).detach().max(0)[0]
        q_target = reward_ts + (gamma * q_target_next * (1 - done_ts))

        # Return priority
        return torch.abs(q_target - q_expected).detach().cpu().numpy()


    def step(self, state, action, reward, next_state, done, alpha, beta):
        """Save experience in replay memory and train the local dqnetwork

        Parameters
        ----------
        state : array-like (37 elements)
            Current state (observation) of the game.
        action : int
            Action taken to reach next state
        reward : float
            Reward returned by moving from state to next_state
        next_state : array-like (37 elements)
            Next state reached by action from (current) state
        done : bool
            Indicates if the episode has finished
        alpha : float
            Prioritization exponent required to calculate sampling probabilities
        beta : float
            Importance sampling weights exponent
        """
        # Get the priority of an experience
        priority = self.get_priority(
            state, action, reward, next_state, done, self.config.gamma)
 
        # Save the experience extended by its priority in the replay buffer
        # To avoid 0 priorities, a small number (1e-4) is added.
        self.memory.add(state, action, reward, next_state, done, priority+1e-4)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random 
            # subset and learn
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample(alpha)
                self.learn(experiences, self.config.gamma, beta)


    def act(self, state, eps=0.0):
        """Returns an action for given a state as per current policy.
        
        Parameters
        ----------
            state : array_like
                Current state (observation) of the game.
            eps : float 
                epsilon, for epsilon-greedy action selection

        Returns
        -------
            The action for a given state as per current policy (local dqnetwork)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get action values of local dqnetwork
        self.dqnetwork_local.eval()
        with torch.no_grad():
            action_values = self.dqnetwork_local(state)
        self.dqnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            # Best action according to current policy
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            # Random action
            return np.random.choice(np.arange(self.action_size)).item()


    def learn(self, experiences, gamma, beta):
        """Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences : (Tuple[torch.Tensor])
            Tuple of (s, a, r, s', done) tuples 
        gamma : float
            Discount factor
        beta : float
            Importance sampling weights exponent
        """
        states, actions, rewards, next_states, dones, probs = experiences

        ## Compute and minimize the loss
        # Get maximum predicted Q values for next states from target model
        Q_targets_next = self.dqnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.dqnetwork_local(states).gather(1, actions)

        # Calculate importance sampling weights (1 / (N * P(i)))^β
        is_weight = torch.pow((1 / (self.config.buffer_size * probs)), beta)
        # Normalize weights by 1 / max w(i) so that they only scale the 
        # update downwards.
        is_weight = torch.div(is_weight, is_weight.max(axis=0).values)
        
        # Compute loss
        loss = F.mse_loss(is_weight*Q_expected, is_weight*Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Update weights of local network
        self.optimizer.step()       
        # Update target network
        self.soft_update(
            self.dqnetwork_local, self.dqnetwork_target, self.config.tau)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model : PyTorch model
            Model from which weights will be copied
        target_model : PyTorch model
            Model to which weights will be copied
        tau : float
            Interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Parameters
        ----------
        action_size : int
            Dimension of each action
        buffer_size : int
            Maximum size of buffer
        batch_size : int
            Size of each training batch
        device : int
            GPU or CPU
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", 
            field_names=["state", "action", "reward", "next_state", "done"])
        self.priority = np.full((buffer_size,), 0.0, dtype=float)
        self.device = device
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory.
        
        Parameters
        ----------
        state : array-like (37 elements)
            Current state (observation) of the game.
        action : int
            Action taken to reach next state
        reward : float
            Reward returned by moving from state to next_state
        next_state : array-like (37 elements)
            Next state reached by action from (current) state
        done : bool
            Indicates if the episode has finished
        priority : float
            Priority of the experience
        """
        # update priority buffer
        if len(self) == len(self.priority):
            # If capacity of the memory is exhaused move all 
            self.priority[:-1], self.priority[-1] = self.priority[1:], priority
        else:
            # Append priority if capacity is not yet exhausted
            self.priority[len(self)] = priority

        # update experience buffer        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, alpha):
        """Randomly sample a batch of experiences from memory.
        
        Samples are not uniformly drawn from the replay memory but according
        to the Prioritized Experience Replay approach.

        Parameters
        ----------
        alpha : float
            Prioritization exponent required to calculate sampling probabilities

        Returns
        -------
            Batch-size expriences as a tuple of (stacked) batch-sized states, 
            actions, rewards, next_states, dones and probabilities.     
        """
        
        # Caluclate the sampling probabilities P(i) = p(i) / Σp(k) for all
        # entries in the replay buffer
        probs = np.power(self.priority[:len(self)], alpha) / \
                         np.sum(np.power(self.priority[:len(self)], alpha))
        
        # Sample a subset of experiences from the replay buffer based on the 
        # probability distribution probs
        idx = np.random.choice(list(range(len(self))), 
                               size=self.batch_size, 
                               replace=False, p=probs)
        experiences = [self.memory[i] for i in idx] 

        # Stack the components of all batch-size experiences including their 
        # respective probabilities 
        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        probs = torch.from_numpy(
            probs[idx].reshape(self.batch_size, 1)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones, probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
