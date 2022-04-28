import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


"""
Follows the codes here:
https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning
"""


class DeepQNetwork(nn.Module):
    """
    Deep Q Network (DQN) implementation for my environment.
    """
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims  # dimension of the observation
        self.fc1_dims = fc1_dims  # fully connected layer dimension
        self.fc2_dims = fc2_dims  # fully connected layer dimension
        self.n_actions = n_actions  # number of feasible actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # fc layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # fc layer 2
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)  # fc layer 3

        self.DQN = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),  # fc layer 1
            nn.ReLU(),  # activation
            nn.Linear(fc1_dims, fc2_dims),  # fc layer 2
            nn.ReLU(),  # activation
            nn.Linear(fc2_dims, n_actions),  # fc layer 3
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # optimizer
        self.loss = nn.MSELoss()  # mean square loss

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        actions = self.DQN(state)

        return actions


class Agent(object):
    """
    The DQN agent.
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon greedy
        self.eps_min = eps_end  # minimum epsilon
        self.eps_dec = eps_dec  # epsilon decay step size
        self.lr = lr  # learning rate
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size  # memory size
        self.batch_size = batch_size  # batch size
        self.mem_cntr = 0  # memory counter
        self.iter_cntr = 0  # iteration counter

        # the network
        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        # memory
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        # epsilon greedy exploration
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # not enough memory to learn from
        if self.mem_cntr < self.batch_size:
            return

        # clear the gradient
        self.Q_eval.optimizer.zero_grad()

        # current memory size
        max_mem = min(self.mem_cntr, self.mem_size)

        # generate batch indices
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # generate batches
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
            self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
            self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
            self.terminal_memory[batch]).to(self.Q_eval.device)

        # calculate the Q values
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # loss back propagate and optimize
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # increase the counter and decrease the epsilon
        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
