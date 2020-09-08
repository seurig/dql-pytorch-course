from deep_q_learning import DeepQAgent
import gym
import numpy as np
from plotting import plot_learning_curve
from tqdm import tqdm

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
n_games_avg = 100
epsilon_history = []

plot_filename = 'plots/cartpole-dql.png'

env = gym.make('CartPole-v1')

agent = DeepQAgent(epsilon, epsilon_decay, epsilon_min, alpha, gamma, state_space_dims, action_space_dims, hidden_layer_dims, batch_size)

for i in tqdm(range(n_games)):
    state = env.reset()
    score = 0
    done = False

    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, state_, done)
        score += reward
        state = state_

    scores.append(score)
    
    if i % 100 == 0: 
        print(f'episode: {i}\t 100 game avg score: {np.mean(scores[-100:]):.2f}\t epsilon: {agent.epsilon:.2f}')
    agent.learn()
env.close()

plot_learning_curve(scores, n_games_avg, agent.epsilon_history, plot_filename)

    
