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
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# Create folders if not exists
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'runs/training_{timestamp}'
image_dir = f'images/training_{timestamp}'
checkpoint_dir = 'checkpoints'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

# For TensorBoard
writer = SummaryWriter(log_dir) 

class ExperienceBuffer:
    def __init__(self, initial_frame, trace_length=4):
        self.trace_length = trace_length
        self.buffer = deque(maxlen=trace_length) # maxlen enables automatic deletion of oldest frame
        
        for _ in range(self.trace_length): # Append trace_length-times the current frame
            self.buffer.append(initial_frame)
    
    def add_frame(self, frame):
        self.buffer.append(frame)
    
    def get_stacked_frames(self):
        stacked_frames = np.concatenate(list(self.buffer), axis=-1)
        stacked_frames = np.transpose(stacked_frames, (2, 0, 1))
        return stacked_frames # (trace_length * channels, height, width)


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
    def __init__(self, replay_size, batch_size, action_dim):
        self.replay_buffer = ReplayMemory(replay_size)
        self.batch_size = batch_size
        self.gamma = 0.95
        self.learningRate = 0.0001
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.999977 # Calculated, depends on amount of episodes (100.000)
        self.target_network_update_frequency = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Select CUDA or CPU
        self.q_network = self.create_model().to(self.device)
        self.target_network = self.create_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learningRate)
        self.episode_loss = 0
        self.action_dim = action_dim
        self.min_memories = 1000

    def create_model(self):
        model = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=6, stride=2),  # 4 Frames, RGB (= 12 channels)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Linear(64 * 8 * 8, 512), # Calculated
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )
        
        return model

    def update_online_network(self):
        if len(self.replay_buffer) < self.min_memories: # Return if the ReplayMemory doesn't have enough memories yet
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.get_memories(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device) # Adds batch dim (batch_size, channels, height, width)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device) # Adds batch dim (batch_size, channels, height, width)
        dones = torch.FloatTensor(dones).to(self.device)
    
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

        self.episode_loss += loss.item()

    def select_action(self, state):
        # Exploration vs. Exploitation
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # PyTorch needs (1, channels, height, width)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def train(self):
        for _ in range(2): # For each step the agent does the network will be trained two times
            self.update_online_network()
        
        if len(self.replay_buffer) % self.target_network_update_frequency: # Update the target network every update_frequency-steps (memories made)
             self.target_network.load_state_dict(self.q_network.state_dict())

    def create_checkpoint(self, filepath, episode, completions):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer.memories),
            'replay_size': self.replay_buffer.memories.maxlen,
            'epsilon': self.epsilon,
            'episode': episode,
            'completions': completions
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            replay_size = checkpoint['replay_size']
            self.replay_buffer = ReplayMemory(replay_size)
            self.replay_buffer.memories = deque(checkpoint['replay_buffer'], maxlen=replay_size)
            self.epsilon = checkpoint['epsilon']
            episode = checkpoint['episode']
            completions = checkpoint['completions']

            return episode, completions
        
        else:
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

# Running main

if __name__ == '__main__':

    trace_length = 4 # Images for experience buffer
    replay_size = 100000 # Memory amount (number of memories (steps, each: current 4 frames & latest 3 + new frame)) for replay buffer (needs to be adjusted to fit RAM-size)
    batch_size = 32 # Amount of memories to be used per training-step
    saveimagesteps = 0 # 0 = no images will be saved, e.g. 2 = every 2 steps an image will be saved
    resume_episode = 0
    completions = 0
    checkpoint_interval = 100

    mission = 'missions/mobchase_single_agent.xml'
    port = 9000
    server = '127.0.0.1'
    port2 = None
    server2 = None
    episodes = 100000
    episode = 0
    role = 0
    episodemaxsteps = 200 # Change according to map later
    saveimagesteps = 0
    resync = 0
    experimentUniqueId = 'test1'

    if server2 is None:
        server2 = server

    xml = Path(mission).read_text()
    env = malmoenv.make()

    env.init(xml, port,
             server=server,
             server2=server2, port2=port2,
             role=role,
             exp_uid=experimentUniqueId,
             episode=episode, resync=resync)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # Number of actions the agent can perform
    # print("state_dim / height", state_dim)
    # print("action_dim / number actions", action_dim)

    # Agent creation
    mc_agent = Agent(replay_size, batch_size, action_dim)

    # Load checkpoint if exists
    try:
        resume_episode, completions = mc_agent.load_checkpoint(checkpoint_path)
        resume_episode += 1 # Start with the next episode
        print(f"Loaded checkpoint. Starting at episode: {resume_episode}")
    except FileNotFoundError:
        print("No checkpoint found. Starting training.")


    for episode in tqdm(range(resume_episode, episodes), desc="Episodes", position=0):
        print("reset " + str(episode))

        # Initial observation and creation ExperienceBuffer
        obs = env.reset()
        experience_buffer = ExperienceBuffer(obs) # Will be recreated every episode

        episode_reward = 0
        mc_agent.episode_loss = 0
        steps = 0
        done = False

        with tqdm(total=episodemaxsteps if episodemaxsteps > 0 else None, desc="Episode steps", position=1, leave=False) as step_bar:
            while not done and (episodemaxsteps <= 0 or steps < episodemaxsteps):
                steps += 1

                state = np.copy(experience_buffer.get_stacked_frames())

                # Select and perform action (random or via model)
                action = mc_agent.select_action(state)
                next_obs, reward, done, info = env.step(action)
                
                episode_reward += reward

                # print("reward: " + str(reward))
                # print("done: " + str(done))
                # print("obs: " + str(next_obs))
                # print(next_obs.size)
                # print("info" + info)
                
                # Update the observation: add the next_obs to the frame stack 
                experience_buffer.add_frame(next_obs)
                # print("experience_buffer", experience_buffer.get_stacked_frames())

                # Now save the experience to the memory
                mc_agent.replay_buffer.add_memory((state, action, reward, experience_buffer.get_stacked_frames(), done))
                # print("memories", mc_agent.replay_buffer.memories)
                
                # Test: Save images
                if saveimagesteps > 0 and steps % saveimagesteps == 0:
                    h, w, d = env.observation_space.shape
                    img = Image.fromarray(obs.reshape(h, w, d))
                    img.save(f'{image_dir}/image_{episode}_{steps}.png')

                # Train the network
                mc_agent.train()

                # Update tqdm bar
                step_bar.update(1)

                time.sleep(2) # Turn off for training / decrease
        
        # Decrease epsilon after each episode
        if mc_agent.epsilon > mc_agent.epsilon_end:
            mc_agent.epsilon *= mc_agent.epsilon_decay

        if done == True:
            completions += 1
            writer.add_scalar('Reward per completion', episode_reward, completions)
        
        # Tracing with TensorBoard
        writer.add_scalar('Epsilon', mc_agent.epsilon, episode)
        writer.add_scalar('Reward', episode_reward, episode)
        writer.add_scalar('Reward/step per episode', episode_reward / steps, episode)
        writer.add_scalar('Steps', steps, episode)
        writer.add_scalar('Average loss', mc_agent.episode_loss	/ steps * 2 , episode) # *2 due to two times training per step
        writer.add_scalar('Completions', completions, episode)

        # Create checkpoint
        if episode % checkpoint_interval == 0:
            mc_agent.create_checkpoint(checkpoint_path, episode, completions)

    env.close()
    writer.close()