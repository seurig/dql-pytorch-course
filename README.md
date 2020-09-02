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

- probabilistic transitions
- ![equation](https://latex.codecogs.com/gif.latex?\sum_{s',r}p(s',r|s,a)=1)
- ![equation](https://latex.codecogs.com/gif.latex?r(s,a)=\sum{r}\sum{p(s',r|s,a)})

- expected return (G): is series of rewards agent expects over time
- ![equation](https://latex.codecogs.com/gif.latex?G_{t}=R_{t&plus;1}&plus;R_{t&plus;2}&plus;R_{t&plus;3}&plus;,...,&plus;R_{T})
- episode: discrete period of gameplay characterized by state, action and reward
- terminal state has no reward
- non-episodic tasks would have infinite episodes -> infinite reward => reward discount (gamma, ![equation](https://latex.codecogs.com/gif.latex?0\leq\gamma\leq1))
- the bigger gamma the more far-sighted the agent acts often (![equation](https://latex.codecogs.com/gif.latex?0.95\leq\gamma\leq0.99))
- expected return with reward discounting (![equation](https://latex.codecogs.com/gif.latex?G_{t}=R_{t&plus;1}&plus;\gamma&space;R_{t&plus;2}&plus;\gamma^{2}R_{t&plus;3}=\sum_{k=0}^{\infty&space;}\gamma^{k}R_{t&plus;k&plus;1}))
- or ![equation](https://latex.codecogs.com/gif.latex?G_{t}=R_{t&plus;1}&plus;\gamma&space;G_{t&plus;1})
- policy (pi): mapping of states to an action

# Value Functions
- states and state/action pairs have values / future rewards
- every policy has a value function for each state (v) and action (q)
- those obey the Bellman equation: recursive relationship between value function and value function of resulting state
- ![equation](https://latex.codecogs.com/gif.latex?v_{\pi}(s)=\sum_{a}\pi(a,s)\sum_{s',r}p(s',r|s,a)[r&plus;\gamma&space;v_{\pi}(s')])

- policies can be ranked by comparing v and q
- Bellman optimality equations: recursive relationship between optimal value function and optimal action value function
- ![equation](https://latex.codecogs.com/gif.latex?v_{*}(s)=max_{a}\sum_{s',r}p(s',r|s,a)[r&plus;\gamma&space;v_{*}(s')])
- ![equation](https://latex.codecogs.com/gif.latex?q_{*}(s,a)=\sum_{s',r}p(s',r|s,a)[r&plus;\gamma&space;maxq_{*}(s',a')])
- to solve Bellman equation you need state transition probabilities either explicitly (dynamic programming/model based learning) or through trial and error (e.g. Q-Learning/model free learning)

# Explore-Exploit Dilemma
- agent never knows whether approximation of enviroment is correct
- explore or exploit model
## Optimistic Inital Values
- give agent overly optimistic values for state value functions
- "disappointment" makes agent explore environment and find an optimal policy
## Epsilon-greedy
- epsilon: fraction of time agent uses for exploration instead of exploitation
- epsilon decay over time, but must stay finite (in training)

# Temporal Difference Learning
- updating agent after each time step vs. after episode end