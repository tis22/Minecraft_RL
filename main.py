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
from agent import ExperienceBuffer, ReplayMemory, Agent

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