# dql-pytorch-course
This repository contains all files created when attending the PyTorch for DQL Udemy course by Phil Tabor.
https://www.udemy.com/course/deep-q-learning-from-paper-to-code

# Reinforcement Learning
- interaction between agent and environment
- action is chosen by agent based on state of environment
- environment changes and returns reward and new state

- state space: set of all possible states
- usefull vs. classification when not all information is present and novel way of solving problem is needed (edge-cases)

- agent uses algorithm (e.g. Q-Learning) to maximize reward

- actions: discrete vs. continuous action spaces
- action space: set of all possible actions
- discrete -> Q-Learning, continuous -> Actor-Critic-Models

# Markov Decision Processes
- state depends only on previous state and action
![equation](https://latex.codecogs.com/gif.latex?%5Csum_%7Bs%27%2C%20r%7D%20p%28s%27%2C%20r%20%7C%20s%2C%20a%29%20%3D%201)
![equation](https://latex.codecogs.com/gif.latex?\sum_{s',&space;r}&space;p(s',&space;r&space;|&space;s,&space;a)&space;=&space;1)