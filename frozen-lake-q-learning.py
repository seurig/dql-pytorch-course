import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from frozen_lake_q_learning_agent import Agent

env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0/
scores = []
avg_100_scores = []

alpha = 0.001
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
n_games = 500000

state_space = np.arange(16)
action_space = np.arange(4)

agent = Agent(alpha, gamma, epsilon, epsilon_decay, epsilon_min, state_space, action_space)

for i in tqdm(range(n_games)):
    state = env.reset()
    done = False
    score = 0
    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)
        #env.render()
        agent.learn(state, action, reward, state_)
        state = state_
        score += reward
    scores.append(score)
    agent.epsilon = agent.epsilon_min if agent.epsilon - agent.epsilon_decay < agent.epsilon_min else agent.epsilon - agent.epsilon_decay
    agent.epsilon_history.append(agent.epsilon)
    if i % 100 == 0: avg_100_scores.append(np.mean(scores[-100:]))

env.close()

plt.plot(np.arange(len(avg_100_scores)), avg_100_scores)
plt.xlabel('batch number')
plt.ylabel('mean score')
plt.title('100 game score average')
plt.savefig('plots/frozen-lake-q-learning.png')
plt.close()

plt.plot(np.arange(len(agent.epsilon_history)), agent.epsilon_history)
plt.xlabel('batch number')
plt.ylabel('epsilon')
plt.title('epsilon history')
plt.savefig('plots/frozen-lake-q-learning-epsilon.png')
plt.close()