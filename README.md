# Neural processes for partially observable gridworld

# POMDP gridworld

The simulator implements a customizable partially observable markov decision process gridworld navigation simulator, where the agent needs to find the goal in a grid by updating its beliefs in a bayesian manner. 

The environment is built by modifying the [minigrid](https://github.com/Farama-Foundation/Minigrid) library, which provides a graphical user interface. The agent uses a model-based reinforcement algorithm that encourages moving towards the agents belief of where the goal is, presented as a probability distribution. Through navigating the environment, the agent updates its beliefs on where the goal state is, prioritizing areas with higher density.

The agent is parameterized by the modes of the belief state; the locations where the agent beliefs goals are. 

# Sampler

The repository provides a sampler class to sample the a dataset of user trajectories and the belief states.

# Training

The [training.ipynb](./training.ipynb) jupyter notebook demonstrates how to sample trajectories, and construct datasets to train an attentive gaussian neural process classifier to predict the trajectories, and a couple of evaluation metrics to measure the quality of the predictions.

# TODO

Training loop for sequential neural processes.