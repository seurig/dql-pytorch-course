# random action for frozen lake environment from gym
# 0 reward per step, 1 for goal
# agent may slide

# 1000 games
# plot win% trailing 10 games

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

wins = []

for _ in range(1000):
    env.reset()
    done = False
    while done is False:
        #env.render()
        observation, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    wins.append(reward)
env.close()

def moving_average(arr, window):
    return np.convolve(arr, np.ones(window), 'valid') / window

wins_trailing_10_games = moving_average(np.array(wins), 10) * 100
plt.plot(np.arange(len(wins_trailing_10_games)), wins_trailing_10_games)
plt.xlabel('batch number')
plt.ylabel('win rate (%)')
plt.title('10 game moving average of win rate')
plt.savefig('plots/frozen-lake.png')
