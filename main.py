import malmoenv
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim


# Select CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

num_actions = 4 # Change to dynamic later

def create_model():
    model = nn.Sequential(
        nn.Conv2d(12, 32, kernel_size=6, stride=2),  # 4 Frames, RGB (= 12 channels)
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=6, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Linear(64 * 8 * 8, 512), # Calculated
        nn.ReLU(),
        nn.Linear(512, num_actions)
    )
    
    return model






# Running main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/mobchase_single_agent.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None, help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=5, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets - default 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print("state_dim / height", state_dim)
    print("action_dim / number actions", action_dim)



    for i in tqdm(range(args.episodes), desc="Episodes", position=0):
        print("reset " + str(i))
        obs = env.reset()

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            steps += 1
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action) # obs is a vector which is the image
            print("reward: " + str(reward))
            print("done: " + str(done))
            print("obs: " + str(obs))
            print(obs.size)
            print("info" + info)

            # Test: Save images
            h, w, d = env.observation_space.shape
            img = Image.fromarray(obs.reshape(h, w, d))
            img.save('images/image' + str(i) + '_' + str(steps) + '.png')
            time.sleep(2)

    env.close()