import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def start_tetris():
    opt = get_args()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size,agent_id=0)
    env2 = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size,agent_id=1)

    while True:
        next_steps = env.get_next_states()  # all possible states
        next_steps2 = env2.get_next_states()

        next_actions, next_states = zip(*next_steps.items())  # all possible actions and states of the current piece
        next_actions2, next_states2 = zip(*next_steps2.items()) # all possible actions

        next_states = torch.stack(next_states)
        next_states2 = torch.stack(next_states2)


        if torch.cuda.is_available():
            next_states = next_states.cuda()
            next_States2 = next_states2.cuda()

        # if random_action:  # epsilon greedy
        index = randint(0, len(next_steps) - 1)
        index2 = randint(0, len(next_steps2) - 1)

        next_state = next_states[index, :]
        action = next_actions[index]
        next_state2 = next_states2[index2, :]
        action2 = next_actions2[index2]


        reward, done, _  = env.step(action, render=True)
        reward2, done2, _ = env2.step(action2, render=True)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
            next_state2 = next_state2.cuda()

        # replay_memory.append([state, reward, next_state, done])  # replay memory for DQN training
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        if done2:
            final_score2 = env2.score
            final_tetrominoes2 = env2.tetrominoes
            final_cleared_lines2 = env2.cleared_lines
            state2 = env2.reset()
            if torch.cuda.is_available():
                state2 = state2.cuda()
        if not done:
            state = next_state
        if not done2:
            state2 = next_state2


if __name__ == "__main__":
    start_tetris()