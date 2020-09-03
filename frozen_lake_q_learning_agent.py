import numpy as np

class Agent():
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min, state_space, action_space):
        self.Q = {}
        for state in state_space: 
            self.Q[state] = {}
            for action in action_space:
                self.Q[state][action] = 0 

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_history = [self.epsilon]
        self.action_space = action_space

    def act(self, state):
        if self.epsilon > np.random.random(): # random action
            action = np.random.choice(self.action_space)
        else: # greedy action
            action = np.argmax(list(self.Q[state].values()))
        return action

    def learn(self, state, action, reward, state_):
        # Q(s,a) = Q'(s,a) + alpha(r + gamma * max (Q(s', amax)) - Q'(s,a))
        Q_ = self.Q[state][action]
        maxQs_ = np.amax(list(self.Q[state_].values()))
        self.Q[state][action] = Q_ + self.alpha * (reward + self.gamma * maxQs_ - Q_)