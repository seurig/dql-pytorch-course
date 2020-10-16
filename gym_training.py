from agents import DuelingDoubleDeepQAgent
from nn_structures import DuelingNeuralNet
import gym
from gym import wrappers
from plotting import plot_learning_curve
import numpy as np
from tqdm import tqdm

game_name = 'CartPole-v1'
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
alpha = 0.0001
gamma = 0.99
update_target_net = 250
weights_file = f'checkpoints/{game_name}-dueling_double_dqn.pt'
n_games = 1500
batch_size = 32
max_memory_size = 50000
evaluation = False

scores, steps_array, epsilon_history = [], [], []
n_score_avg = 100
n_steps = 0
best_score = -np.inf

plot_filename = f'plots/{game_name}-dueling_double_dql.png'

env = gym.make(game_name)

#env = wrappers.Monitor(env, 'videos/video', video_callable=lambda episode_id: True, force=True)

state_space_dims, action_space_dims = env.observation_space.shape, env.action_space.n

DQN = DuelingNeuralNet(state_space_dims, action_space_dims, alpha, weights_file)
target_DQN = DuelingNeuralNet(state_space_dims, action_space_dims, alpha, weights_file)

agent = DuelingDoubleDeepQAgent(epsilon, epsilon_decay, epsilon_min, gamma, state_space_dims, action_space_dims, DQN, target_DQN, update_target_net, batch_size, max_memory_size)

for i in tqdm(range(n_games)):

    state = env.reset()
    score = 0
    done = False

    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)

        if evaluation is False:
            agent.memory.remember(state, action, reward, state_, done)
            agent.learn()
        
        score += reward
        state = state_
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)
    avg_score = np.mean(scores[-n_score_avg:])   

    print(f'episode: {i}\t {n_score_avg} game avg score: {avg_score:.2f}\t epsilon: {agent.epsilon:.2f}\t steps: {n_steps}')
    if avg_score > best_score: 
        if evaluation is False:
            agent.DQN.save_weights()
        best_score = avg_score
    epsilon_history.append(agent.epsilon)
env.close()

plot_learning_curve(steps_array, scores, n_score_avg, epsilon_history, plot_filename)

    
