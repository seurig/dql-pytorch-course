import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_dims, fc1_dims, output_dims, lr):
        super(NeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, output_dims)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):       
        layer1 = F.relu(self.fc1(data))
        layer2 = self.fc2(layer1)

        return layer2     

class DeepQAgent():
    def __init__(self, epsilon, epsilon_decay, epsilon_min, alpha, gamma, state_space_dims, action_space_dims, hidden_layer_dims, batch_size):
        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_history = [epsilon]
        
        self.state_memory, self.action_memory, self.reward_memory, self.state__memory, self.terminal_memory = [], [], [], [], []

        self.batch_size = batch_size
        self.gamma = gamma
        self.action_space_dims = action_space_dims

        self.DQN = NeuralNet(state_space_dims, hidden_layer_dims, action_space_dims, alpha)

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min 
        self.epsilon_history.append(self.epsilon)

    def remember(self, state, action, reward, state_, done):
        self.state_memory.append(state)
        self.action_memory.append(np.eye(self.action_space_dims, dtype=np.bool)[action])
        self.reward_memory.append(reward)
        self.state__memory.append(state_)
        self.terminal_memory.append(done)

    def sample_memory(self):
        if len(self.state_memory) >= self.batch_size:
            idc = np.random.choice(np.arange(len(self.state_memory)), self.batch_size)
            return np.array(self.state_memory)[idc], np.array(self.action_memory)[idc], np.array(self.reward_memory)[idc], \
                np.array(self.state__memory)[idc], np.array(self.terminal_memory)[idc]
        else: return None

    def learn(self):
        mem_sample = self.sample_memory()
        if mem_sample is not None:
            states, actions, rewards, states_, _ = mem_sample
            self.DQN.optimizer.zero_grad()

            states = T.tensor(states, dtype=T.float).to(self.DQN.device)
            actions = T.tensor(actions).to(self.DQN.device)
            rewards = T.tensor(rewards).to(self.DQN.device)
            states_ = T.tensor(states_, dtype=T.float).to(self.DQN.device)

            Q_pred = self.DQN.forward(states)
            Q_next = self.DQN.forward(states_)

            Q_pred = T.masked_select(Q_pred, actions)
            Q_next, _ = T.max(Q_next, dim=-1)
            
            # Q(s,a) = Q'(s,a) + alpha(r + gamma * max (Q(s', amax)) - Q'(s,a))
            # MSE = mse(r + gamma * max (Q(s', amax)), Q(s,a))
            Q_target = rewards + self.gamma * Q_next
            
            cost = self.DQN.loss(Q_pred, Q_target).to(self.DQN.device)
            cost.backward()
            self.DQN.optimizer.step()
            self.decay_epsilon()      

    def act(self, state):
        if self.epsilon > np.random.random():
            return np.random.choice(np.arange(self.action_space_dims))
        else:
            state = T.tensor(state, dtype=T.float).to(self.DQN.device)
            return T.argmax(self.DQN.forward(state)).item()