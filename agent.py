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
        """
        Initialize the Experience Buffer with a given initial frame.
        ---------

        Args:
            initial_frame (np.array): The first frame to initialize the buffer.
            height (int): Height of the input frames.
            width (int): Width of the input frames.
            channels (int): Number of channels in the input frames (e.g. RGB = 3).
            trace_length (int): The number of frames to stack in the buffer (default is 4).
        ---------

        Returns:
            None.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.trace_length = trace_length
        # maxlen enables automatic deletion of oldest frame
        self.buffer = deque(maxlen=trace_length) 
        reshaped_obs = initial_frame.reshape((self.height, self.width, self.channels))
        
        # Append trace_length-times the current frame
        for _ in range(self.trace_length): 
            self.buffer.append(reshaped_obs)
    
    def add_frame(self, frame):
        """
        Add a new frame to the experience buffer.
        ---------

        Args:
            frame (np.array): The new frame to be added to the buffer.
        ---------

        Returns:
            None.
        """
        reshaped_obs = frame.reshape((self.height, self.width, self.channels))
        self.buffer.append(reshaped_obs)
    
    def get_stacked_frames(self):
        """
        Get the stacked frames from the buffer, combined into a single tensor.
        ---------

        Args:
            None.
        ---------

        Returns:
            np.array: The stacked frames with shape (trace_length * channels, height, width).
        """
        stacked_frames = np.concatenate(list(self.buffer), axis=-1)
        stacked_frames = np.transpose(stacked_frames, (2, 0, 1))
        return stacked_frames # (trace_length * channels, height, width)


class ReplayMemory:
    def __init__(self, replay_size):
        """
        Initialize the replay memory with a specified size.
        ---------

        Args:
            replay_size (int): The maximum number of memories to store in the replay buffer.
        ---------

        Returns:
            None.
        """
        self.memories = deque(maxlen=replay_size)
    
    def add_memory(self, transition):
        """
        Add a transition (state, action, reward, next_state, done) to the replay memory.
        ---------

        Args:
            transition (tuple): A tuple containing (state, action, reward, next_state, done).
        ---------

        Returns:
            None.
        """
        self.memories.append(transition)
    
    def get_memories(self, batch_size):
        """
        Sample a batch of memories from the replay buffer.
        ---------

        Args:
            batch_size (int): The number of memories to sample.
        ---------

        Returns:
            tuple: A tuple containing the sampled states, actions, rewards, next_states and dones.
        """
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
        """
        Initialize the Agent with necessary parameters and networks.
        ---------

        Args:
            replay_size (int): Maximum number of memories in the replay buffer (default is 150000).
            batch_size (int): The size of the batches used for training (default is 128).
            action_dim (int): The number of possible actions (default is 4).
            gamma (float): Discount factor for future rewards (default is 0.97).
            learning_rate (float): Learning rate for optimizer (default is 0.0001).
            epsilon (float): Initial epsilon for exploration-exploitation trade-off (default is 1.0).
            epsilon_end (float): The minimum epsilon value (default is 0.1).
            epsilon_decay (float): The decay rate for epsilon (default is 0.999954).
            target_network_update_frequency (int): Frequency for updating the target network (default is 2000).
            device (str): The device to run the model on (default is "cuda" if available).
            min_memories (int): The minimum number of memories required for training (default is 1000).
        ---------

        Returns:
            None.
        """
        self.replay_buffer = ReplayMemory(replay_size)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.learningRate = learning_rate
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        # Epsilon decay is calculated, depends on amount of episodes (100.000)
        self.epsilon_decay = epsilon_decay 
        self.target_network_update_frequency = target_network_update_frequency
        # Select CUDA or CPU
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu")) 
        self.min_memories = min_memories
        self.q_network = self.create_model().to(self.device)
        self.target_network = self.create_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learningRate)
        self.episode_loss = 0
        self.steps_made = 0

    def create_model(self):
        """
        Create the Q-network model.
        ---------

        Args:
            None.
        ---------

        Returns:
            nn.Sequential: The PyTorch model representing the Q-network.
        """
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
        """
        Update the online Q-network using a batch of memories from the replay buffer.
        ---------

        Args:
            None.
        ---------

        Returns:
            None.
        """
        # Return if the ReplayMemory doesn't have enough memories yet
        if len(self.replay_buffer.memories) < self.min_memories: 
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.get_memories(self.batch_size)
        
        # Adds batch dim (batch_size, channels, height, width)
        states = torch.FloatTensor(np.array(states)).to(self.device) 
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        # Adds batch dim (batch_size, channels, height, width)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) 
        dones = torch.FloatTensor(dones).to(self.device)
    
        # Computed for all items in the memory-batch
        # Predicted Q-Values for current state (online-network)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) 

        # Next actions predicted by online-network
        next_actions = self.q_network(next_states).argmax(dim=1) 

        # Q-Values in the next state (target-network)
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1) 

        # Estimated future reward from taking action a in state s 
        q_targets = rewards + self.gamma * next_q_values * (1 - dones) 

        # Actual learning
        # Loss (difference between prediction and target)
        loss = nn.MSELoss()(q_values, q_targets) 
        self.optimizer.zero_grad() # Reset gradients
        loss.backward() # Backpropagation
        self.optimizer.step() # Update weights

        self.episode_loss += loss.item()

    def select_action(self, state):
        """
        Select an action based on the current state using epsilon-greedy policy.
        ---------

        Args:
            state (np.array): The current state.
        ---------
        
        Returns:
            int: The selected action.
        """
        # Exploration vs. Exploitation
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            # PyTorch needs (1, channels, height, width)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def train(self):
        """
        Train the agent by updating the online network and periodically updating the target network.
        ---------

        Args:
            None.
        ---------

        Returns:
            None.
        """
        # For each step the agent does the network will be trained two times
        for _ in range(2): 
            self.update_online_network()
        
        # Update the target network every update_frequency-steps (memories made)
        if self.steps_made % self.target_network_update_frequency == 0: 
             self.target_network.load_state_dict(self.q_network.state_dict())

    def create_checkpoint(self, checkpoint_path, memories_path, episode, completions, base_name):
        """
        Create and save a checkpoint of the agent's state, including the Q-network, target network, optimizer and replay memory.
        ---------

        Args:
            checkpoint_path (str): The path to save the checkpoint.
            memories_path (str): The path to save the replay memory.
            episode (int): The current episode number.
            completions (int): The number of completions achieved.
            base_name (str): The base name for checkpoint files.
        ---------

        Returns:
            None.
        """
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
        """
        Load a checkpoint of the agent's state, including the Q-network, target network, optimizer and replay memory.
        ---------

        Args:
            checkpoint_path (str): The path to the checkpoint file.
            memories_path (str, optional): The path to the replay memory file (default is None).
        ---------

        Returns:
            tuple: A tuple containing the episode number, completions and base name.
        """
        try:
            checkpoint_metadata = torch.load(checkpoint_path, map_location=self.device)

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
                    raise FileNotFoundError(f"Replay buffer file not found: {memories_path}")

            if 'memories' in locals():
                del memories
            gc.collect()
            
            return episode, completions, base_name

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def download_model(self, filepath):
        """
        Download a pre-trained model from a URL.
        ---------

        Args:
            filepath (str): The path to save the downloaded model.
        ---------

        Returns:
            None.
        """
        url = 'https://drive.google.com/file/d/1srxlOZYg-oNERTyVKHRy0trTDAsoMRWn/view?usp=sharing'
        model_path = gdown.download(url, filepath, fuzzy=True, use_cookies=False, quiet=False)
