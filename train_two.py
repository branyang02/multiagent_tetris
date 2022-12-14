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

torch.set_printoptions(threshold=10_000)


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
    parser.add_argument("--replay_memory_size", type=int, default=30_000,  # default = 30_000
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="double_trained_new")

    args = parser.parse_args()
    return args

def combine_states(state1, state2):
    cat_states1, cat_states2 = [], [] 
    for s1 in state1:
        for s2 in state2:
            cat_states1.append(torch.cat((s1, s2), 0))
            cat_states2.append(torch.cat((s2, s1), 0))
    return torch.stack(cat_states1), torch.stack(cat_states2)

def average_predictions(predictions, n1, n2):
    averaged_predictions = []
    for action in range(n1):
        averaged_predictions.append(torch.mean(predictions[action*n2:action*(n2+1)]))
    return torch.cuda.FloatTensor(averaged_predictions)

def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    # if os.path.isdir(opt.log_path):
    #     shutil.rmtree(opt.log_path)
    # os.makedirs(opt.log_path)

    writer = SummaryWriter(opt.log_path)
    writer2 = SummaryWriter(opt.log_path + "_2")

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size, agent_id=0)
    env2 = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size, agent_id=1)

    model = DeepQNetwork()
    model2 = DeepQNetwork()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)  # Adam Optimizer
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=opt.lr)  # Adam Optimizer

    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()

    state = env.reset()
    state2 = env2.reset()

    state_temp = state
    state = torch.cat((state, torch.tensor([state2[2]])))
    state2 = torch.cat((state2, torch.tensor([state_temp[2]])))

    if torch.cuda.is_available():
        model.cuda()
        model2.cuda()
        state = state.cuda()
        state2 = state2.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    replay_memory2 = deque(maxlen=opt.replay_memory_size)
    b2b1, b2b2 = 0, 0

    epoch = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        next_steps2 = env2.get_next_states()

        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

        epsilon2 = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

        u = random()
        u2 = random()

        random_action = u <= epsilon
        random_action2 = u2 <= epsilon2
        
        next_actions, next_states = zip(*next_steps.items())  # all possible actions and states of the current piece
        next_actions2, next_states2 = zip(*next_steps2.items())  # all possible actions and states of the current piece
        # num_states1, num_states2 = len(next_states), len(next_states2)
        
        next_states = torch.stack(next_states)
        m1 = torch.nn.ConstantPad1d((0,1), state2[2])
        next_states = m1(next_states)

        next_states2 = torch.stack(next_states2)
        m2 = torch.nn.ConstantPad1d((0,1), state[2])
        next_states2 = m2(next_states2)

        # next_states, next_states2 = combine_states(next_states, next_states2)

        if torch.cuda.is_available():
            next_states = next_states.cuda()
            next_states2 = next_states2.cuda()

        model.eval()
        model2.eval()

        with torch.no_grad():
            predictions = model(next_states)[:, 0]
            # predictions = average_predictions(predictions, num_states1, num_states2)
            predictions2 = model2(next_states2)[:, 0]
            # predictions2 = average_predictions(predictions2, num_states2, num_states1)

        model.train()
        model2.train()

        if random_action:  # epsilon greedy
            index = randint(0, len(predictions) - 1)
        else:
            index = torch.argmax(predictions).item()

        if random_action2:  # epsilon greedy
            index2 = randint(0, len(predictions2) - 1)
        else:
            index2 = torch.argmax(predictions2).item()

        next_state = next_states[index, :]
        action = next_actions[index]
        # action = next_actions[index // num_states2]

        next_state2 = next_states2[index2, :]
        action2 = next_actions2[index2]
        # action2 = next_actions2[index2 // num_states1]

        height_diff_reward = (next_state2[3].item() - next_state[3].item()) / 10
        reward, done, cleared_lines = env.step(action, render=False)
        reward += height_diff_reward
        reward2, done2, cleared_lines2 = env2.step(action2, render=False)
        reward2 -= height_diff_reward

        # if a player dies, then the other player gets rewarded
        if done ^ done2:
            if done:
                reward2 += 10
            if done2:
                reward += 10
        
        if (not done and not done2) and cleared_lines > 0:
            if cleared_lines > 1:
                env2.garbage(cleared_lines, b2b1)
            if b2b1 and cleared_lines == 4:
                reward += b2b1
            b2b1 = b2b1 + 1 if cleared_lines == 4 else 0


        if (not done and not done2) and cleared_lines2 > 0:
            if cleared_lines2 > 1:
                env.garbage(cleared_lines2, b2b2)
            if b2b2 and cleared_lines2 == 4:
                reward2 += b2b2
            b2b2 = b2b2 + 1 if cleared_lines2 == 4 else 0

        if torch.cuda.is_available():
            next_state = next_state.cuda()
            next_state2 = next_state2.cuda()
    
        replay_memory.append([state, reward, next_state, done])  # replay memory for DQN training
        replay_memory2.append([state2, reward2, next_state2, done2]) # replay memory for DQN training

        if done or done2:
            b2b1, b2b2 = 0, 0
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()

            final_score2 = env2.score
            final_tetrominoes2 = env2.tetrominoes
            final_cleared_lines2 = env2.cleared_lines
            state2 = env2.reset()

            # state_temp = state
            # state = torch.cat((state, state2))
            # state2 = torch.cat((state2, state_temp))
            state_temp = state
            state = torch.cat((state, torch.tensor([state2[2]])))
            state2 = torch.cat((state2, torch.tensor([state_temp[2]])))

            if torch.cuda.is_available():
                state = state.cuda()
                state2 = state2.cuda()

        if not done and not done2:
            state = next_state
            state2 = next_state2
            continue


        if len(replay_memory) < opt.replay_memory_size / 10 and len(replay_memory2) < opt.replay_memory_size / 10:
            print("collecting experience replay")
            continue

        epoch += 1

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        batch2 = sample(replay_memory2, min(len(replay_memory2), opt.batch_size))
        state_batch2, reward_batch2, next_state_batch2, done_batch2 = zip(*batch2)
        state_batch2 = torch.stack(tuple(state2 for state2 in state_batch2))
        reward_batch2 = torch.from_numpy(np.array(reward_batch2, dtype=np.float32)[:, None])
        next_state_batch2 = torch.stack(tuple(state2 for state2 in next_state_batch2))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

            state_batch2 = state_batch2.cuda()
            reward_batch2 = reward_batch2.cuda()
            next_state_batch2 = next_state_batch2.cuda()

        q_values = model(state_batch)
        q_values2 = model2(state_batch2)

        model.eval()
        model2.eval()

        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
            next_prediction_batch2 = model2(next_state_batch2)

        model.train()
        model2.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        y_batch2 = torch.cat(
            tuple(reward2 if done2 else reward2 + opt.gamma * prediction2 for reward2, done2, prediction2 in
                  zip(reward_batch2, done_batch2, next_prediction_batch2)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        optimizer2.zero_grad()
        loss2 = criterion2(q_values2, y_batch2)
        loss2.backward()
        optimizer2.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

        print("Epoch2: {}/{}, Action2: {}, Score2: {}, Tetrominoes2 {}, Cleared lines2: {}".format(
            epoch,
            opt.num_epochs,
            action2,
            final_score2,
            final_tetrominoes2,
            final_cleared_lines2))
        writer2.add_scalar('Train/Score', final_score2, epoch - 1)
        writer2.add_scalar('Train/Tetrominoes', final_tetrominoes2, epoch - 1)
        writer2.add_scalar('Train/Cleared lines', final_cleared_lines2, epoch - 1)

        print("---------------------------------------------")

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model2, "{}/tetris_{}".format(opt.saved_path + "_2", epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))
    torch.save(model2, "{}/tetris".format(opt.saved_path+"_2"))


if __name__ == "__main__":
    opt = get_args()
    train(opt)