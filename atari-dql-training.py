from deep_q_agent import DeepQAgent
from nn_structures import ConvNeuralNet
from atari_wrapper import make_environment
from plotting import plot_learning_curve
import numpy as np
from tqdm import tqdm

game_name = 'PongNoFrameskip-v4'
epsilon = 1.0
epsilon_decay = 1e-5
epsilon_min = 0.1
alpha = 0.0001
gamma = 0.99
update_target_net = 1000
weights_file = f'checkpoints/{game_name}-dqn.pt'
n_games = 10 #500
batch_size = 32
max_memory_size = 20000 # 50000
evaluation = False

scores, steps_array, epsilon_history = [], [], []
n_games_avg = 100
n_steps = 0
best_score = -np.inf

plot_filename = f'plots/{game_name}-dql.png'

env = make_environment(game_name)
state_space_dims, action_space_dims = env.observation_space.shape, env.action_space.n

DQN = ConvNeuralNet(state_space_dims, action_space_dims, alpha, weights_file)
target_DQN = ConvNeuralNet(state_space_dims, action_space_dims, alpha, weights_file)

agent = DeepQAgent(epsilon, epsilon_decay, epsilon_min, gamma, state_space_dims, action_space_dims, DQN, target_DQN, update_target_net, batch_size, max_memory_size)

for i in tqdm(range(n_games)):
    state = env.reset()
    score = 0
    done = False

    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)

        if evaluation:
            agent.memory.remember(state, action, reward, state_, int(done))
            agent.learn()
        
        score += reward
        state = state_
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)
    avg_score = np.mean(scores[-n_games_avg:])   

    if i % n_games_avg == 0: 
        print(f'episode: {i}\t {n_games_avg} game avg score: {avg_score:.2f}\t epsilon: {agent.epsilon:.2f}\t steps: {n_steps}')
        if avg_score > best_score: 
            if evaluation is False:
                agent.DQN.save_weights()
            best_score = avg_score
    epsilon_history.append(agent.epsilon)
env.close()

plot_learning_curve(steps_array, scores, n_games_avg, epsilon_history, plot_filename)

    
