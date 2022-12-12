import argparse
from random import randint
import torch
import cv2
from src.tetris import Tetris
from src.tetris2 import Tetris2


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = get_args()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env2 = Tetris2(width=opt.width, height=opt.height, block_size=opt.block_size)

    # Obtain the trained model
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        print("torch.cuda is available")
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)

    model.eval()

    env.reset()
    env2.reset()

    if torch.cuda.is_available():
        model.cuda()
    
    while True:
        next_steps = env.get_next_states()
        next_steps2 = env2.get_next_states()

        next_actions, next_states = zip(*next_steps.items())  # all possible actions and states of the current piece
        next_actions2, next_states2 = zip(*next_steps2.items()) # all possible actions

        next_states = torch.stack(next_states)
        next_states2 = torch.stack(next_states2)

        if torch.cuda.is_available():
            next_states = next_states.cuda()
            next_States2 = next_states2.cuda()

        # model prediction
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]

        # random action from dumb agent
        index2 = randint(0, len(next_steps2) - 1)
        next_state2 = next_states2[index2, :]
        action2 = next_actions2[index2]

        #step
        reward, done, cleared_lines = env.step(action, render=True)
        reward2, done2, cleared_lines2 = env2.step(action2, render=True)

        # print(env.board)
        # print(cleared_lines)
        print(env.current_pos)
        print(env2.height)
        if cleared_lines > 1:
            print("resetting")
            print(env2.garbage())
        # exit()

        if torch.cuda.is_available():
            # next_state = next_state.cuda()
            next_state2 = next_state2.cuda()

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
        # if not done:
        #     state = next_state
        if not done2:
            state2 = next_state2