import argparse
import agents as Agents
import nn_structures as NN_DQL
import gym
from gym import wrappers
from plotting import plot_learning_curve
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='DQL Training in gym environment')
parser.add_argument('-env_name', type=str, default='CartPole-v1', help='which environment to train on')
parser.add_argument('-algorithm', type=str, default='DuelingDoubleDeepQAgent', help='which learning algorithm to use')
parser.add_argument('-epsilon', type=float, default=1.0, help='starting epsilon value')
parser.add_argument('-epsilon_decay', type=float, default=0.9995, help='number epsilon is multiplied by every iteration')
parser.add_argument('-epsilon_min', type=float, default=0.01, help='min epsilon value')
parser.add_argument('-alpha', type=float, default=0.0001, help='learning rate')
parser.add_argument('-gamma', type=float, default=0.99, help='future experience importance reduction factor')
parser.add_argument('-update_target_net', type=int, default=250, help='after how many iterations to update target nn')
parser.add_argument('-n_games', type=int, default=1500, help='how many games to play')
parser.add_argument('-batch_size', type=int, default=32, help='batch size for nn training')
parser.add_argument('-max_memory_size', type=int, default=50000, help='max number datasets in memory')
parser.add_argument('-evaluation', type=bool, default=False, help='no training')

args = parser.parse_args()

weights_file = f'checkpoints/{args.env_name}-{args.algorithm}.pt'

epsilon, epsilon_decay, epsilon_min = args.epsilon, args.epsilon_decay, args.epsilon_min
if args.evaluation is True: epsilon, epsilon_decay, epsilon_min = 0, 0, 0

scores, steps_array, epsilon_history = [], [], []
n_score_avg = 100
n_steps = 0
best_score = -np.inf

plot_filename = f'plots/{args.env_name}-{args.algorithm}.png'

env = gym.make(args.env_name)

state_space_dims, action_space_dims = env.observation_space.shape, env.action_space.n

if 'Dueling' in args.algorithm:
    nn_ = getattr(NN_DQL, 'DuelingNeuralNet')
else:
    nn_ = getattr(NN_DQL, 'NeuralNet')

DQN = nn_(state_space_dims, action_space_dims, args.alpha, weights_file)
target_DQN = nn_(state_space_dims, action_space_dims, args.alpha, weights_file)

if args.evaluation is True: 
    DQN.load_weights()
    target_DQN.load_weights()

agent_ = getattr(Agents, args.algorithm)
agent = agent_(epsilon, epsilon_decay, epsilon_min, args.gamma, state_space_dims, action_space_dims, DQN, target_DQN, args.update_target_net, args.batch_size, args.max_memory_size)

for i in tqdm(range(args.n_games)):

    state = env.reset()
    score = 0
    done = False

    while done is False:
        action = agent.act(state)
        state_, reward, done, _ = env.step(action)

        if args.evaluation is False:
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
        if args.evaluation is False:
            agent.DQN.save_weights()
        best_score = avg_score
    epsilon_history.append(agent.epsilon)
env.close()

plot_learning_curve(steps_array, scores, n_score_avg, epsilon_history, plot_filename)