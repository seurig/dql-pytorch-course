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
- ![equation](https://latex.codecogs.com/gif.latex?\sum_{s',r}p(s',r|s,a)=1)
- ![equation](https://latex.codecogs.com/gif.latex?r(s,a)=\sum{r}\sum{p(s',r|s,a)})