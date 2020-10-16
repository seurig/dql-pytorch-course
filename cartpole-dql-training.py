from agents import DeepQAgent
from nn_structures import NeuralNet
import gym
import numpy as np
from plotting import plot_learning_curve
from tqdm import tqdm

epsilon = 1
epsilon_decay = 0.9995
epsilon_min = 0.01
alpha = 0.0001
gamma = 0.99
hidden_layer_dims = 128
update_target_net = 250
weights_file = 'checkpoints/cartpole-dqn.pt'
n_games = 1500
batch_size = 32
max_memory_size = 50000

state_space_dims, action_space_dims = [4], 2

scores = []
n_games_avg = 100
avg_scores = []
epsilon_history = []

plot_filename = 'plots/cartpole-dql.png'

env = gym.make('CartPole-v1')

DQN = NeuralNet(state_space_dims, hidden_layer_dims, action_space_dims, alpha, weights_file)
target_DQN = NeuralNet(state_space_dims, hidden_layer_dims, action_space_dims, alpha, weights_file)

agent = DeepQAgent(epsilon, epsilon_decay, epsilon_min, gamma, state_space_dims, action_space_dims, DQN, target_DQN, update_target_net, batch_size, max_memory_size)

for i in tqdm(range(n_games)):
    state = env.reset()
    score = 0
    done = False

    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)
        agent.memory.remember(state, action, reward, state_, done)
        agent.learn()
        score += reward
        state = state_

    scores.append(score)
    avg_score = np.mean(scores[-n_games_avg:])
    max_score = np.amax(avg_scores) if len(avg_scores) > 0 else -np.inf
    avg_scores.append(avg_score)
    epsilon_history.append(agent.epsilon)

    if i % n_games_avg == 0: 
        print(f'episode: {i}\t {n_games_avg} game avg score: {avg_score:.2f}\t epsilon: {agent.epsilon:.2f}')
        if avg_score > max_score: agent.DQN.save_weights()
    
env.close()

plot_learning_curve(np.arange(len(scores)), scores, n_games_avg, epsilon_history, plot_filename)

    
