import malmoenv
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Select CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


num_actions = 4 # Change to dynamic later (action_dim = env.action_space.n)
trace_length = 4 # Images for experience buffer
replay_size = 100000 # Memory amount (number of memories) for replay buffer (needs to be adjusted to fit RAM-size)
batch_size = 32 # Amount of memories to be used per training-step



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


class ExperienceBuffer:
    def __init__(self, initial_frame, trace_length=4):
        self.trace_length = trace_length
        self.buffer = deque(maxlen=trace_length) # maxlen enables automatic deletion of oldest frame
        
        for _ in range(self.trace_length): # Append trace_length-times the current frame
            self.buffer.append(initial_frame)
    
    def add_frame(self, frame):
        self.buffer.append(frame)
    
    def get_stacked_frames(self):
        return np.concatenate(list(self.buffer), axis=-1)  # Stack along channel-axis (height, width, channels)


class ReplayMemory:
    def __init__(self, replay_size):
        self.memories = deque(maxlen=replay_size)
    
    def add_memory(self, transition):
        self.memories.append(transition)
    
    def get_memories(self, batch_size):
        batch = random.sample(self.memories, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, replay_size, batch_size):
        self.replay_buffer = ReplayMemory(replay_size)
        self.batch_size = batch_size
        self.gamma = 0.95
        self.learningRate = 0.0001
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.999977 # Calculated, depends on amount of episodes (100.000)
        self.target_network_update_frequency = 1000
        self.q_network = create_model().to(device)
        self.target_network = create_model().to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learningRate)

    def update_online_network(self):
        if len(self.replay_buffer) < batch_size: # Return if the ReplayMemory doesn't have enough memories yet
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.get_memories(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
    
        # Computed for all items in the memory-batch
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Predicted Q-Values for current state (online-network)
        next_actions = self.q_network(next_states).argmax(dim=1) # Next actions predicted by online-network
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1) # Q-Values in the next state (target-network)
        q_targets = rewards + self.gamma * next_q_values * (1 - dones) # Estimated future reward from taking action a in state s 

        # Actual learning
        loss = nn.MSELoss()(q_values, q_targets) # Loss (difference between prediction and target)
        self.optimizer.zero_grad() # Reset gradients
        loss.backward() # Backpropagation
        self.optimizer.step() # Update weights

    def select_action(self, state):
        # Exploration vs. Exploitation
        if random.random() <= self.epsilon:
            return random.randrange(num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def train(self):
        for _ in range(2): # For each step the agent does the network will be trained two times
            self.update_online_network()
        
        if len(self.replay_buffer) % self.target_network_update_frequency: # Update the target network every update_frequency-steps (memories made)
             self.target_network.load_state_dict(self.q_network.state_dict())


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

        experience_buffer = ExperienceBuffer(obs)
        mc_agent = Agent(replay_size, batch_size)

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            steps += 1
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action) # obs is a vector which is the image (next state)
            print("reward: " + str(reward))
            print("done: " + str(done))
            print("obs: " + str(obs))
            print(obs.size)
            print("info" + info)


            mc_agent.replay_buffer.add_memory(((experience_buffer.get_stacked_frames), action, reward, obs, done))
            print("memories", mc_agent.replay_buffer.memories)

            experience_buffer.add_frame(obs)
            print("experience_buffer", experience_buffer.get_stacked_frames())


            # Test: Save images
            h, w, d = env.observation_space.shape
            img = Image.fromarray(obs.reshape(h, w, d))
            img.save('images/image' + str(i) + '_' + str(steps) + '.png')
            time.sleep(2)
        
        # Decrease epsilon after each episode
        # if mc_agent.epsilon > mc_agent.epsilon_end:
        #    mc_agent.epsilon *= mc_agent.epsilon_decay

    env.close()