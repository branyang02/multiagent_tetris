import argparse
import torch
import cv2
from src.tetris import Tetris
from src.tetris2 import Tetris2
import time

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="double_trained")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        print("torch.cuda is available")
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
        model2 = torch.load("{}/tetris".format(opt.saved_path+ "_2"))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
    
    model.eval()
    model2.eval()

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env2 = Tetris2(width=opt.width, height=opt.height, block_size=opt.block_size)
    
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


    # out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
    #                       (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    
    b2b1, b2b2 = 0, 0
    while True:
        # time.sleep(0.1)
        next_steps = env.get_next_states()
        next_steps2 = env2.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_actions2, next_states2 = zip(*next_steps2.items())
        
        # cat_states1, cat_states2 = [], [] #torch.empty(0, 4)
        # for s1 in next_states:
        #     for s2 in next_states2:
        #         cat_states1.append(torch.cat((s1, s2), 0))
        #         cat_states2.append(torch.cat((s2, s1), 0))
        # cat_states1 = torch.stack(cat_states1)
        # cat_states2 = torch.stack(cat_states2)
        next_states = torch.stack(next_states)
        m1 = torch.nn.ConstantPad1d((0,1), state2[2])
        next_states = m1(next_states)

        next_states2 = torch.stack(next_states2)
        m2 = torch.nn.ConstantPad1d((0,1), state[2])
        next_states2 = m2(next_states2)

        # next_states = torch.stack(next_states)
        # next_states2 = torch.stack(next_states2)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
            next_states2 = next_states2.cuda()
            # cat_states1 = cat_states1.cuda()
            # cat_states2 = cat_states2.cuda()

        predictions = model(next_states)[:, 0]
        predictions2 = model2(next_states2)[:, 0]
        index = torch.argmax(predictions).item()
        index2 = torch.argmax(predictions2).item()
        action = next_actions[index]
        action2 = next_actions2[index2]
        next_state = next_states[index, :]
        next_state2 = next_states2[index2, :]

        # print(action)
        _, done, cleared_lines = env.step(action, render=True, video=None)
        _, done2, cleared_lines2 = env2.step(action2, render=True, video=None)

        if (not done and not done2) and cleared_lines > 0:
            if cleared_lines > 1:
                env2.garbage(cleared_lines, b2b1)
            b2b1 = b2b1 + 1 if cleared_lines == 4 else 0


        if (not done and not done2) and cleared_lines2 > 0:
            if cleared_lines2 > 1:
                env.garbage(cleared_lines2, b2b2)
            b2b2 = b2b2 + 1 if cleared_lines2 == 4 else 0

        if torch.cuda.is_available():
            next_state = next_state.cuda()
            next_state2 = next_state2.cuda()

        if not done and not done2:
            state = next_state
            state2 = next_state2
            continue

        if done and done2:
            print("they somehow managed to lose at the same time")
            break
        if done:
            # out.release()
            print("player 2 wins")
            break
        if done2:
            # out2.release()
            print("player 1 wins")
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
