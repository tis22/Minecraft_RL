import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import gdown
import pickle
import gc

class ExperienceBuffer:
    def __init__(self, initial_frame, height, width, channels, trace_length=4):
        self.height = height
        self.width = width
        self.channels = channels
        self.trace_length = trace_length
        self.buffer = deque(maxlen=trace_length) # maxlen enables automatic deletion of oldest frame
        reshaped_obs = initial_frame.reshape((self.height, self.width, self.channels))
        
        for _ in range(self.trace_length): # Append trace_length-times the current frame
            self.buffer.append(reshaped_obs)
    
    def add_frame(self, frame):
        reshaped_obs = frame.reshape((self.height, self.width, self.channels))
        self.buffer.append(reshaped_obs)
    
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
    def __init__(self, 
                 replay_size=150000, 
                 batch_size=128, 
                 action_dim=4, 
                 gamma=0.97, 
                 learning_rate=0.0001, 
                 epsilon=1.0, 
                 epsilon_end=0.1, 
                 epsilon_decay=0.999954, # 0.1 will be reached at 50000 episodes
                 target_network_update_frequency=2000, 
                 device=None, 
                 min_memories=1000):

        self.replay_buffer = ReplayMemory(replay_size)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.learningRate = learning_rate
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay # Calculated, depends on amount of episodes (100.000)
        self.target_network_update_frequency = target_network_update_frequency
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu")) # Select CUDA or CPU
        self.min_memories = min_memories
        self.q_network = self.create_model().to(self.device)
        self.target_network = self.create_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learningRate)
        self.episode_loss = 0
        self.steps_made = 0

    def create_model(self):
        model = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=6, stride=2),  # 4 Frames, RGB (= 12 channels)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), # Calculated
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )
        
        return model

    def update_online_network(self):
        if len(self.replay_buffer.memories) < self.min_memories: # Return if the ReplayMemory doesn't have enough memories yet
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.get_memories(self.batch_size)
        
        states = torch.FloatTensor(np.array(states)).to(self.device) # Adds batch dim (batch_size, channels, height, width)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) # Adds batch dim (batch_size, channels, height, width)
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
        
        if self.steps_made % self.target_network_update_frequency == 0: # Update the target network every update_frequency-steps (memories made)
             self.target_network.load_state_dict(self.q_network.state_dict())

    def create_checkpoint(self, checkpoint_path, memories_path, episode, completions, base_name):
        try:
            checkpoint_metadata = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_model_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'replay_size': self.replay_buffer.memories.maxlen,
                'epsilon': self.epsilon,
                'episode': episode,
                'completions': completions,
                'steps_made': self.steps_made,
                'base_name': base_name
            }

            torch.save(checkpoint_metadata, checkpoint_path)
            
            with open(memories_path, 'wb') as f:
                pickle.dump(self.replay_buffer.memories, f)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path, memories_path=None):
        try:
            checkpoint_metadata = torch.load(checkpoint_path)

            self.q_network.load_state_dict(checkpoint_metadata['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint_metadata['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_metadata['optimizer_state_dict'])
            self.epsilon = checkpoint_metadata['epsilon']
            episode = checkpoint_metadata['episode']
            completions = checkpoint_metadata['completions']
            self.steps_made = checkpoint_metadata['steps_made']
            base_name = checkpoint_metadata['base_name']

            # Load replay buffer 
            if memories_path is not None:
                replay_size = checkpoint_metadata['replay_size']

                if os.path.isfile(memories_path):
                    with open(memories_path, 'rb') as f:
                        memories = pickle.load(f)
                    self.replay_buffer = ReplayMemory(replay_size)
                    self.replay_buffer.memories = deque(memories, maxlen=replay_size)
                else:
                    raise FileNotFoundError(f"Replay-Buffer-Datei nicht gefunden: {memories_path}")

            if 'memories' in locals():
                del memories
            gc.collect()
            
            return episode, completions, base_name

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def download_model(self, filepath):
        url = ''
        model_path = gdown.download(url, filepath, fuzzy=True, use_cookies=False, quiet=False)
        self.q_network = torch.load(filepath)
