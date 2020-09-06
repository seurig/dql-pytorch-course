from deep_q_learning import DeepQAgent
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_epsilon(history):
    plt.plot(np.arange(len(history)), history)
    plt.xlabel('batch number')
    plt.ylabel('epsilon')
    plt.title('epsilon history')
    plt.savefig('plots/cartpole-q-learning-epsilon.png')
    plt.close()

def plot_scores(scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('batch number')
    plt.ylabel('mean score')
    plt.title('100 game score average')
    plt.savefig('plots/cartpole-q-learning-score.png')
    plt.close()

epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01
alpha = 0.001
gamma = 0.99
hidden_layer_dims = 128
n_games = 10000
batch_size = 32

state_space_dims, action_space_dims = 4, 2

scores = []
avg_100_scores = []
epsilon_history = []

env = gym.make('CartPole-v0')

agent = DeepQAgent(epsilon, epsilon_decay, epsilon_min, alpha, gamma, state_space_dims, action_space_dims, hidden_layer_dims, batch_size)

for i in tqdm(range(n_games)):
    state = env.reset()
    score = 0
    epsilon_history.append(agent.epsilon)
    done = False

    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, state_, done)
        score += reward
        state = state_

    scores.append(score)
    
    if i % 100 == 0: 
        avg_100_scores.append(np.mean(scores[-100:]))
        print(f'episode: {i}\t 100 game avg score: {avg_100_scores[-1]:.2f}\t epsilon: {agent.epsilon:.2f}')
    agent.learn()
    agent.decay_epsilon()
env.close()

plot_scores(avg_100_scores)
plot_epsilon(agent.epsilon_history)

    
