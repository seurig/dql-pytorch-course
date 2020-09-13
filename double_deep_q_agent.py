import torch as T
import numpy as np

from agent_memory import AgentMemory

class DoubleDeepQAgent():
    def __init__(self, epsilon, epsilon_decay, epsilon_min, \
                        gamma, \
                        state_space_dims, action_space_dims, \
                        DQN, target_DQN, update_target_net, \
                        batch_size, max_memory_size):

        self.epsilon = epsilon
        self.epsilon_max = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.action_space_dims = action_space_dims

        self.DQN = DQN
        self.target_DQN = target_DQN
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.tau = update_target_net
        self.weight_replace_counter = 0

        self.memory = AgentMemory(state_space_dims, max_memory_size)

    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min     

    def replace_DQN(self):
        self.weight_replace_counter += 1
        if self.weight_replace_counter % self.tau == 0:
            self.target_DQN.load_state_dict(self.DQN.state_dict())

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.DQN.optimizer.zero_grad()
        self.replace_DQN()

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        states = T.tensor(states).to(self.DQN.device)
        actions = T.tensor(actions).to(self.DQN.device)
        rewards = T.tensor(rewards).to(self.DQN.device)
        states_ = T.tensor(states_).to(self.DQN.device)
        dones = T.tensor(dones).to(self.DQN.device)

        batch_indices = np.arange(self.batch_size)

        Q_pred = self.DQN.forward(states)[batch_indices, actions]

        Q_argmax = T.argmax(self.DQN.forward(states_), dim=1)
        
        Q_next = self.target_DQN.forward(states_)[batch_indices, Q_argmax]
        Q_next[dones] = 0.0
        
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
            state = T.tensor([state], dtype=T.float).to(self.DQN.device)
            return T.argmax(self.DQN.forward(state)).item()