import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_epsilon(history):
    plt.plot(np.arange(len(history)), history)
    plt.xlabel('batch number')
    plt.ylabel('epsilon')
    plt.title('epsilon history')
    plt.savefig('plots/frozen-lake-q-learning-epsilon.png')
    plt.close()

def plot_scores(scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('batch number')
    plt.ylabel('mean score')
    plt.title('100 game score average')
    plt.savefig('plots/frozen-lake-q-learning.png')
    plt.close()

from frozen_lake_q_learning_agent import Agent

env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0/
scores = []
avg_100_scores = []

alpha = 0.001
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.9999995
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
        agent.learn(state, action, reward, state_, done)
        state = state_
        score += reward
    scores.append(score)
    if i % 100 == 0: avg_100_scores.append(np.mean(scores[-100:]))
    if i % 5000 == 0: 
        print(f'episode: {i}\t 100 game avg score: {np.mean(scores[-100:]):.2f}\t epsilon: {agent.epsilon:.2f}')
env.close()

plot_scores(avg_100_scores)
plot_epsilon(agent.epsilon_history)