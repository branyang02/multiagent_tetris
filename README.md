# [PYTORCH] Multiplayer Tetris

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

# test multiplayer model vs single-player model
python multi-trained_vs_single-trained.py
```

# Models

## Muti-player Model
The optimal model for multiplayer is located in the following directories:
- /double_trained_new
- /double_trained_new_2

## Single-Player Model
The model for single-player Tetris is located in the directory:
- /trained_models

## Training Results

### Train Score:
![Train Score](https://github.com/branyang02/multiagent_tetris/blob/main/LeakyReLU_50k_results/Train_Score.svg)

### Number of Cleared Lines:
![Number of Cleared Lines](https://github.com/branyang02/multiagent_tetris/blob/main/LeakyReLU_50k_results/Train_Cleared_lines.svg)

### Number of Tetrominoes:
![Number of Tetrominoes](https://github.com/branyang02/multiagent_tetris/blob/main/LeakyReLU_50k_results/Train_Tetrominoes.svg)

## Experiments
We trained our multi-agent model and a single-agent model to play multiplayer Tetris and compared their performance using the following metrics:

|    | Multi-Agent Model | Single-Agent Model |
|----|--------------------|--------------------|
|Average Score | 730.41 | 613.93 |
|Average Lines Cleared | 35.90 | 29.03 |
|Win Rate| 87.2% | 12.8% |
- **Average Score**: This metric represents the average score achieved by the agent over a certain number of games. A higher score indicates that the agent was able to clear more lines and survive longer in the game. The results show that the multi-agent model had an average score of 730.41, which is higher than the single-agent model's average score of 613.93. This suggests that the multi-agent model is better at achieving a higher score in the game.

- **Average Lines Cleared**: This metric represents the average number of lines cleared by the agent over a certain number of games. A higher number of lines cleared indicates that the agent is better at clearing lines and surviving longer in the game. The results show that the multi-agent model had an average of 35.90 lines cleared per game, which is higher than the single-agent model's average of 29.03 lines cleared per game. This suggests that the multi-agent model is better at clearing lines and surviving longer in the game.

- **Win Rate**: This metric represents the percentage of games won by the agent over a certain number of games. A higher win rate indicates that the agent is better at outlasting its opponents. The results show that the multi-agent model had a win rate of 87.2%, while the single-agent model had a win rate of 12.8%.



# Hyperparameter Tuning

The following hyperparameters were used to train the multiplayer agent:

- `width`: The common width for all images, set to 10.
- `height`: The common height for all images, set to 20.
- `block_size`: Size of a block, set to 30.
- `batch_size`: The number of images per batch, set to 512.
- `lr`: Learning rate, set to 1e-3.
- `gamma`: Discount factor, set to 0.99.
- `initial_epsilon`: Initial value of epsilon for epsilon-greedy exploration, set to 1.
- `final_epsilon`: Final value of epsilon for epsilon-greedy exploration, set to 1e-3.
- `num_decay_epochs`: Number of epochs for epsilon decay, set to 2000.
- `num_epochs`: Total number of training epochs, set to 3000.
- `save_interval`: Number of epochs between model saves, set to 1000.
- `replay_memory_size`: Size of the replay memory, set to 30,000.



# Authors
* Alan Zheng
* Brandon Yang
* Boheng Mu

