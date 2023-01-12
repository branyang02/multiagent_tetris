# [PYTORCH] DQN for Multiplayer Tetris

# Multiplayer Tetris with Reinforcement Learning

This project is an extension of https://github.com/uvipen/Tetris-deep-Q-learning-pytorch, a library that uses deep Q-learning to train a single player Tetris agent. Our work builds on this by developing a multiplayer Tetris AI using Multi-Agent Reinforcement Learning (MARL) techniques. The goal of the project is to train the agent to consider the state of its opponent's board while making decisions, similarly to how a human player would behave.

Our work demonstrated that introducing parameter sharing between different agents in Tetris allows agents to accomplish state-of-the-art multiplayer Tetris strategies. 

# Demo
![](https://github.com/arxk9/multiagent_tetris/blob/main/demo/trained_agents.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

- Python 3.6+
- PyTorch 1.7+
- numpy 
- matplotlib
- cv2
- PIL

### Installing

A step by step series of examples that tell you how to get a development env running

```bash
# Clone the repository
git clone https://github.com/arxk9/multiagent_tetris.git

# navigate to the folder
cd multiagent_tetris

# train multiplayer model
python train_two.py

# test multiplyer model
python trained_vs_trained.py
```

# Models
The optimal model is located in /double_trained_new and /double_trained_new_2

Here are the results:
![](https://github.com/branyang02/multiagent_tetris/blob/main/LeakyReLU_50k_results/Train_Score.svg)


# Authors
* Alan Zheng
* Brandon Yang
* Boheng Mu

