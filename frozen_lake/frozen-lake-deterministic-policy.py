# random action for frozen lake environment from gym
# 0 reward per step, 1 for goal
# agent may slide

# 0 => left
# 1 => down
# 2 => right
# 3 => up

import gym
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self):
        self.directions = {'left':0, 'down':1, 'right':2, 'up':3}

    def act(self, state):
        if state in [0,1,8,9,13,14]: return self.directions['right']
        if state in [2,4,6,10]: return self.directions['down']
        if state in [3]: return self.directions['left']

env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0/

wins = []
avg_win_10_games = []

for i in range(1000):
    agent = Agent()
    observation = env.reset()
    done = False
    #env.render()

    while done is False:
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        #env.render()
    wins.append(reward) # either 0 or 1 when game is done - no reward given before

    if i % 10 == 0:
        avg_win_10_games.append(np.mean(wins[-10:]) * 100)

env.close()

plt.plot(np.arange(len(avg_win_10_games)), avg_win_10_games)
plt.xlabel('batch number')
plt.ylabel('win rate (%)')
plt.title('10 game average of win rate')
plt.savefig('plots/frozen-lake-deterministic.png')