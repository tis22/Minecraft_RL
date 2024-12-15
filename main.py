import malmoenv
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from agent import ExperienceBuffer, Agent
import subprocess
import gdown
import zipfile
from lxml import etree
from threading import Thread, Event

def train():
    xml = Path(mission).read_text()
    env = malmoenv.make()

    env.init(xml, port,
             server=server,
             server2=server2, port2=port2,
             role=role,
             exp_uid=experimentUniqueId,
             episode=episode, resync=resync)
    
    action_dim = env.action_space.n # Number of actions the agent can perform

    # Agent creation
    mc_agent = Agent(replay_size, batch_size, action_dim)

    # Load checkpoint if exists
    try:
        resume_episode, completions, base_name = mc_agent.load_checkpoint(checkpoint_path)
        resume_episode += 1 # Start with the next episode
        print(f"Loaded checkpoint. Starting at episode: {resume_episode}")
    except FileNotFoundError:
        print("No checkpoint found. Starting training.")

        # Create folders if not exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"training_{timestamp}"

    log_dir = f'runs/{base_name}'
    image_dir = f'images/{base_name}'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # For TensorBoard
    writer = SummaryWriter(log_dir) 
    if args.start_tensorboard:
        start_tensorboard(log_dir)

    # Main training loop
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
            mc_agent.create_checkpoint(checkpoint_path, episode, completions, base_name)
        
        # Create permanent checkpoints for evaluation
        if episode % permanent_checkpoint_interval == 0:
            permanent_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth')
            mc_agent.create_checkpoint(permanent_checkpoint_path, episode, completions, base_name)
            
    writer.close()
    env.close()


def evaluate():
    # Agent creation
    mc_agent = Agent(replay_size, batch_size)

    if os.path.isfile(checkpoint_path):
        try:
            mc_agent.load_checkpoint(checkpoint_path)
            print("Loaded model.")
        except Exception as e:
            print(f"Error loading the model: {e}")
            return
    else:
        print("Model does not exist. Downloading from Google Drive.")
        mc_agent.download_model(checkpoint_path)
        try:
            mc_agent.load_checkpoint(checkpoint_path)
            print("Loaded downloaded model.")
        except Exception as e:
            print(f"Error loading the downloaded model: {e}")
            return
    
    xml = Path(mission_eval).read_text()
    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
        
    agent_done_event = Event()
    global_stop_event = Event()

    stop_thread = Thread(target=wait_for_stop, args=(global_stop_event,))
    stop_thread.start()
    threads = [Thread(target=run_evaluate, args=(i, global_stop_event, agent_done_event, xml, mc_agent)) for i in range(number_of_agents)]
    
    [t.start() for t in threads]
    [t.join() for t in threads]

    stop_thread.join()


def run_evaluate(role, global_stop_event, agent_done_event, xml, mc_agent):
    env = malmoenv.make()
    env.init(xml, port,
             server=server,
             server2=server2, port2=port + role,
             role=role,
             exp_uid=experimentUniqueId,
             episode=episode, resync=resync)
    
    # Set actual action dimension
    if role == 0:
        mc_agent.action_dim = env.action_space.n

    # Main evaluate loop
    while not global_stop_event.is_set():
        # Initial observation and creation ExperienceBuffer
        obs = env.reset()
        if role == 0:
            experience_buffer = ExperienceBuffer(obs) # Will be recreated every episode
        steps = 0
        done = False
        total_reward = 0
        
        # One episode
        while not done and not global_stop_event.is_set() and (episodemaxsteps <= 0 or steps < episodemaxsteps):
            steps += 1

            if role == 0: # Agent
                state = experience_buffer.get_stacked_frames()

                # Select and perform action (random or via model)
                action = mc_agent.select_action(state)
                next_obs, reward, done, info = env.step(action)

                # Update the observation: add the next_obs to the frame stack 
                experience_buffer.add_frame(next_obs)

                total_reward += reward
                log(f"Step {steps}: Action={action}, Reward={reward}, Total reward={total_reward}, Done={done}")
            
            elif role == 1:  # Spectator-Agent
                if agent_done_event.is_set():
                    log("Spectator stopping because agent_done_event is set.")
                    break
                action = 0
                next_obs, reward, done, info = env.step(action)

            if done and role == 0:
                agent_done_event.set()
                print("Agent is done. Event set.")

            time.sleep(0.9)

        if role == 0:
            agent_done_event.clear()

        env.close()

def wait_for_stop(global_stop_event):
    input("Press Enter to stop...\n")
    global_stop_event.set()

def log(message):
    print(f'[{role}] {message}')

def start_tensorboard(log_dir=None):

    if not os.path.exists('runs'):
        os.makedirs('runs')
    
    if log_dir is None:
        dirs = [d for d in os.listdir('runs') if d.startswith('training_')]

        if not dirs:
            print("No log directories found.")
            log_url = "XXX"

            try:
                print("Downloading logs from Google Drive.")
                output_path = os.path.join('runs', 'logs.zip')
                gdown.download(log_url, output_path, fuzzy=True, use_cookies=False, quiet=False)
                print("Logs downloaded successfully.")

                # Unzip
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    # Use the name of the folder
                    extracted_folder = zip_ref.namelist()[0]
                    extraction_path = os.path.join('runs', extracted_folder)
                    zip_ref.extractall(extraction_path)
                    print(f"Logs extracted to {extraction_path}.")
                
                # Remove ZIP
                os.remove(output_path)

                log_dir = extraction_path

            except Exception as e:
                print(f"Failed to download logs: {e}")
                return
            
        else:
            dirs.sort(reverse=True) # New ones first
            log_dir = os.path.join('runs', dirs[0])

    if log_dir:
        subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])
        print(f"TensorBoard is running on http://localhost:6006 with logs from {log_dir}")
    else:
        print("No log directory found or specified.")

if __name__ == '__main__':

    checkpoint_dir = 'checkpoints'
    trace_length = 4 # Images for experience buffer

    # Training parameters
    episodes = 100000
    episodemaxsteps = 200 # Change according to map later
    replay_size = 100000 # Memory amount (number of memories (steps, each: current 4 frames & latest 3 + new frame)) for replay buffer (needs to be adjusted to fit RAM-size)
    batch_size = 32 # Amount of memories to be used per training-step
    saveimagesteps = 0 # Training: 0 = no images will be saved, e.g. 2 = every 2 steps an image will be saved
    resume_episode = 0
    completions = 0
    checkpoint_interval = 100
    permanent_checkpoint_interval = 10000

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

    # Check what the user want to do
    parser = argparse.ArgumentParser(description="Train or evaluate the Minecraft agent")
    parser.add_argument('--train', action='store_true', help='Train the Minecraft agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the Minecraft agent')
    parser.add_argument('--tensorboard', action='store_true', help='Start TensorBoard during training')
    parser.add_argument('--tensorboard-only', action='store_true', help='Only start TensorBoard')
    parser.add_argument('--logdir', type=str, default='runs', help='Directory for TensorBoard logs')
    args = parser.parse_args()

    mission = 'missions/mobchase_single_agent.xml'
    mission_eval = 'missions/mobchase_two_agents.xml'
    port = 9000
    server = '127.0.0.1'
    port2 = None
    server2 = None
    episode = 0
    role = 0
    resync = 0
    experimentUniqueId = 'MinecraftMaze'

    if server2 is None:
        server2 = server
    
    try:
        if args.tensorboard_only:
            start_tensorboard(args.logdir)
        elif args.train:
            train()
        elif args.eval:
            evaluate()
        else:
            print("Choose between --train or --eval or --tensorboard-only")
    except KeyboardInterrupt:
        print("\nAborted by user.")
