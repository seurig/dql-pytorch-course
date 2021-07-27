''' Collection of Deep Q Learning agents '''

import numpy as np
import torch as T

from agent_memory import AgentMemory

class Agent():
    ''' Agent base class for class inheritance '''

    def __init__(self, epsilon, epsilon_decay, epsilon_min, \
                        gamma, \
                        state_space_dims, action_space_dims, \
                        DQN, target_DQN, update_target_net, \
                        batch_size, max_memory_size):
        ''' creating Agent with the given parameters:
        - epsilon
        - epsilon_decay
        - epsilon_min
        - gamma
        - state_space_dims
        - action_space_dims
        - DQN
        - target_DQN
        - update_target_net
        - batch_size
        - max_memory_size
        '''

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
        ''' decreasing epsilon forcing more greedy actions '''
        self.epsilon = self.epsilon * self.epsilon_decay \
                if self.epsilon * self.epsilon_decay > self.epsilon_min \
                else self.epsilon_min

    def replace_DQN(self):
        ''' replace target DQN weights with DQN weights '''
        self.weight_replace_counter += 1
        if self.weight_replace_counter % self.tau == 0:
            self.target_DQN.load_state_dict(self.DQN.state_dict())

    def learning_algorithm(self):
        ''' learning algorithm to be implemented after class inheritance '''
        raise NotImplementedError

    def learn(self):
        ''' training Deep Q Network with memories from memory '''
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

        Q_pred, Q_next = self.learning_algorithm(
            states, actions, rewards, states_, dones, batch_indices
            )

        Q_target = rewards + self.gamma * Q_next

        cost = self.DQN.loss(Q_pred, Q_target).to(self.DQN.device)
        cost.backward()
        self.DQN.optimizer.step()

        self.decay_epsilon()

    def act(self, state):
        ''' how agent acts upon given state to be implemented after class inheritance '''
        raise NotImplementedError

class DeepQAgent(Agent):
    ''' Agent for Deep Q Learning '''
    def __init__(self, *args, **kwargs):
        ''' creating using class inheritance '''
        super(DeepQAgent, self).__init__(*args, **kwargs)

    def learning_algorithm(self, states, actions, rewards, states_, dones, batch_indices):
        ''' Deep Q learning algorithm '''
        Q_pred = self.DQN.forward(states)[batch_indices, actions]

        Q_next = self.target_DQN.forward(states_)
        Q_next, _ = T.max(Q_next, dim=1)

        Q_next[dones] = 0.0

        return Q_pred, Q_next

    def act(self, state):
        ''' epsilon greedy action '''
        if self.epsilon > np.random.random():
            return np.random.choice(np.arange(self.action_space_dims))
        else:
            state = T.tensor([state], dtype=T.float).to(self.DQN.device)
            return T.argmax(self.DQN.forward(state)).item()

class DoubleDeepQAgent(Agent):
    ''' Agent for Double Deep Q Learning '''
    def __init__(self, *args, **kwargs):
        ''' creating using class inheritance '''
        super(DoubleDeepQAgent, self).__init__(*args, **kwargs)

    def learning_algorithm(self, states, actions, rewards, states_, dones, batch_indices):
        ''' Double Deep Q learning algorithm '''
        Q_pred = self.DQN.forward(states)[batch_indices, actions]

        Q_argmax = T.argmax(self.DQN.forward(states_), dim=1)

        Q_next = self.target_DQN.forward(states_)[batch_indices, Q_argmax]
        Q_next[dones] = 0.0

        return Q_pred, Q_next

    def act(self, state):
        ''' epsilon greedy action '''
        if self.epsilon > np.random.random():
            return np.random.choice(np.arange(self.action_space_dims))
        else:
            state = T.tensor([state], dtype=T.float).to(self.DQN.device)
            return T.argmax(self.DQN.forward(state)).item()

class DuelingDeepQAgent(Agent):
    ''' Agent for Dueling Deep Q Learning '''
    def __init__(self, *args, **kwargs):
        ''' creating using class inheritance '''
        super(DuelingDeepQAgent, self).__init__(*args, **kwargs)

    def learning_algorithm(self, states, actions, rewards, states_, dones, batch_indices):
        ''' Dueling Deep Q learning algorithm '''
        V_states, A_states = self.DQN.forward(states)
        V_states_, A_states_ = self.target_DQN.forward(states_)
        Q_pred = T.add(V_states, A_states - A_states.mean(dim=1, keepdim=True))[batch_indices, actions]
        Q_next, _ = T.add(V_states_, A_states_ - A_states_.mean(dim=1, keepdim=True)).max(dim=1)

        Q_next[dones] = 0.0

        return Q_pred, Q_next

    def act(self, state):
        ''' epsilon greedy action '''
        if self.epsilon > np.random.random():
            return np.random.choice(np.arange(self.action_space_dims))
        else:
            state = T.tensor([state], dtype=T.float).to(self.DQN.device)
            _, A = self.DQN.forward(state)
            return T.argmax(A).item()

class DuelingDoubleDeepQAgent(Agent):
    ''' Agent for Dueling Double Deep Q Learning '''
    def __init__(self, *args, **kwargs):
        ''' creating using class inheritance '''
        super(DuelingDoubleDeepQAgent, self).__init__(*args, **kwargs)

    def learning_algorithm(self, states, actions, rewards, states_, dones, batch_indices):
        ''' Dueling Double Deep Q learning algorithm '''
        V_states, A_states = self.DQN.forward(states)
        V_states_, A_states_ = self.target_DQN.forward(states_)

        V_eval, A_eval = self.DQN.forward(states_)
        Q_argmax = T.argmax(T.add(V_eval, A_eval - A_eval.mean(dim=1, keepdim=True)), dim=1)

        Q_pred = T.add(
            V_states, A_states - A_states.mean(dim=1, keepdim=True)
                )[batch_indices, actions]
        Q_next = T.add(
            V_states_, A_states_ - A_states_.mean(dim=1, keepdim=True)
                )[batch_indices, Q_argmax]

        Q_next[dones] = 0.0

        return Q_pred, Q_next

    def act(self, state):
        ''' epsilon greedy action '''
        if self.epsilon > np.random.random():
            return np.random.choice(np.arange(self.action_space_dims))
        else:
            state = T.tensor([state], dtype=T.float).to(self.DQN.device)
            _, A = self.DQN.forward(state)
            return T.argmax(A).item()
