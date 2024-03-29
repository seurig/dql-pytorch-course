''' AgentMemory class for use in Reinforcement Learning '''

import numpy as np

class AgentMemory():
    ''' Memory class for saving and sampling Q function data '''

    def __init__(self, state_space_dims, max_size):
        ''' creating agent memory with given:
        - state_space_dims
        - max_size
        '''
        self.max_size = max_size
        self.memory_counter = 0

        self.state_memory = np.zeros(shape=(max_size, *state_space_dims), dtype=np.float32)
        self.action_memory = np.zeros(shape=(max_size), dtype=np.int64)
        self.reward_memory = np.zeros(shape=(max_size), dtype=np.float32)
        self.state__memory = np.zeros(shape=(max_size, *state_space_dims), dtype=np.float32)
        self.terminal_memory = np.zeros(shape=(max_size), dtype=np.bool)

    def remember(self, state, action, reward, state_, done):
        ''' saving Q function data in memory '''

        idx = self.memory_counter % self.max_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.state__memory[idx] = state_
        self.terminal_memory[idx] = done

        self.memory_counter += 1

    def sample(self, batch_size):
        ''' sampling from memory with given batch size '''

        n_data_pairs = min(self.memory_counter, self.max_size)
        idc = np.random.choice(n_data_pairs, batch_size, replace=False)
        return self.state_memory[idc], self.action_memory[idc], self.reward_memory[idc], \
            self.state__memory[idc], self.terminal_memory[idc]
