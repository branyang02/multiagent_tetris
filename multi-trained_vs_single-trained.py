import argparse
import torch
import cv2
from src.tetris import Tetris
from tqdm import tqdm

### Agent trained on multi-agent network vs agent trained on single-agent network ###
# Agent_id 0: multi-agent network
# Agent_id 1: single-agent network

agent_1_score = 0
agent_2_score = 0


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="double_trained_2")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--num_experiment", type=int, default=1000)

    args = parser.parse_args()
    return args


def test(opt):    
    global agent_2_score, agent_1_score
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load("{}/tetris_4000".format(opt.saved_path))
        model2 = torch.load("{}/tetris".format("trained_models"))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
        model2 = torch.load("{}/tetris".format("trained_models"), map_location=lambda storage, loc: storage)

    model.eval()
    model2.eval()

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size, agent_id=0)
    env2 = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size, agent_id=1)

    ## State = [lines_cleared, holes, bumpiness, height]
    state = env.reset()
    state2 = env2.reset()

    # state_temp = state
    state = torch.cat((state, torch.tensor([state2[2]])))  # include bumpiness of the second agent
    
    if torch.cuda.is_available():
        model.cuda()
        model2.cuda()
        state = state.cuda()
        state2 = state2.cuda()

    b2b1, b2b2 = 0, 0

    while True:
        next_steps = env.get_next_states()
        next_steps2 = env2.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_actions2, next_states2 = zip(*next_steps2.items())
        
        next_states = torch.stack(next_states)
        m1 = torch.nn.ConstantPad1d((0,1), state2[2])
        next_states = m1(next_states)

        next_states2 = torch.stack(next_states2)
        
        if torch.cuda.is_available():
            next_states = next_states.cuda()
            next_states2 = next_states2.cuda()

        predictions = model(next_states)[:, 0]
        predictions2 = model2(next_states2)[:, 0]
        index = torch.argmax(predictions).item()
        index2 = torch.argmax(predictions2).item()
        action = next_actions[index]
        action2 = next_actions2[index2]
        next_state = next_states[index, :]
        next_state2 = next_states2[index2, :]

        _, done, cleared_lines = env.step(action, render=False, video=None)
        _, done2, cleared_lines2 = env2.step(action2, render=False, video=None)

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
            # print("player 2 wins")
            agent_2_score += 1
            break
        if done2:
            # print("player 1 wins")
            agent_1_score += 1
            break



if __name__ == "__main__":
    opt = get_args()
    for i in tqdm((range(opt.num_experiment)), desc="Testing...", ascii=False, ncols=75):
        # print()
        test(opt)
    print("agent 1 score: ", agent_1_score)
    print("agent 2 score: ", agent_2_score)
    print("multi-agent trained win rate: ", (agent_1_score / (agent_2_score + agent_1_score)) * 100, "%")