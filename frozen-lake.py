# random action for frozen lake environment from gym
# 0 reward per step, 1 for goal
# agent may slide

# 1000 games
# plot win% for every 10 games

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0/

wins = []
avg_win_10_games = []

for i in range(1000):
    env.reset()
    done = False

    while done is False:
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action) # take a random action

    wins.append(reward) # either 0 or 1 when game is done - no reward given before

    if i % 10 == 0:
        avg_win_10_games.append(np.mean(wins[-10:]) * 100)

env.close()

plt.plot(np.arange(len(avg_win_10_games)), avg_win_10_games)
plt.xlabel('batch number')
plt.ylabel('win rate (%)')
plt.title('10 game average of win rate')
plt.savefig('plots/frozen-lake.png')
