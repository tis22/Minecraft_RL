import malmoenv
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from PIL import Image
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
import re

def train():
    """
    Train the Minecraft agent using Reinforcement Learning.
    Initializes the environment and agent and runs a training loop where the agent interacts with the environment.
    During training, the agent updates its model, saves checkpoints and logs metrics.
    ---------

    Returns:
        None.
    """
    global mission, port, server, server2, port2, role, experimentUniqueId, resume_episode, episode, resync, permanent_checkpoint_interval, batch_size, replay_size
    xml = Path(mission).read_text()
    env = malmoenv.make()

    env.init(xml, port,
             server=server,
             server2=server2, port2=port2,
             role=role,
             exp_uid=experimentUniqueId,
             episode=episode, resync=resync)
    
    h, w, c = env.observation_space.shape
    action_dim = env.action_space.n # Number of actions the agent can perform

    # Agent creation
    mc_agent = Agent(replay_size, batch_size, action_dim)

    # Load checkpoint if exists
    try:
        resume_episode, completions, base_name = mc_agent.load_checkpoint(checkpoint_path, memories_path)
        resume_episode += 1 # Start with the next episode
        print(f"Loaded checkpoint. Starting at episode: {resume_episode}")

        image_dir = f'images/{base_name}'
        for image_file in os.listdir(image_dir):
            # Get episode number from filename
            match = re.match(r'image_(\d+)_\d+\.png', image_file)

            if match:
                episode_number = int(match.group(1))
                # Delete images with episodes >= resume_episode
                if episode_number >= resume_episode:
                    os.remove(os.path.join(image_dir, image_file))

    except FileNotFoundError:
        print("No checkpoint found. Starting training.")

        completions = 0

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

    # Main training loop
    for episode in tqdm(range(resume_episode, episodes), desc="Episodes", position=0):

        # Initial observation and creation ExperienceBuffer
        obs = env.reset()
        experience_buffer = ExperienceBuffer(obs, h, w, c)

        episode_reward = 0
        mc_agent.episode_loss = 0
        steps = 0
        done = False
        current_orientation = 0

        if saveimagesteps > 0:
            img = Image.fromarray(obs.reshape(h, w, c))
            img.save(f'{image_dir}/image_{episode}_{steps}.png')

        with tqdm(total=episodemaxsteps if episodemaxsteps > 0 else None, desc="Episode steps", position=1, leave=False) as step_bar:
            while not done and (episodemaxsteps <= 0 or steps < episodemaxsteps):
                steps += 1

                state = np.copy(experience_buffer.get_stacked_frames())

                # Select and perform action (random or via model)
                action = mc_agent.select_action(state)
                next_obs, reward, done, info = env.step(action)

                if action == 2:  # Rotate right (90°)
                    current_orientation = (current_orientation + 1) % 4
                elif action == 3:  # Rotate left (90°)
                    current_orientation = (current_orientation - 1) % 4

                # Additional reward for moving forward (northwards)
                if action == 0 and current_orientation == 0:
                    reward += 0.15

                # Check if next_obs is valid (not empty or invalid)
                if next_obs is None or next_obs.size == 0:
                    # print(f"Warning: Encountered empty observation at step {steps} during episode {episode}. Skipping this step.")
                    done = True
                    break

                mc_agent.steps_made += 1
                episode_reward += reward

                # print("reward: " + str(reward))
                # print("done: " + str(done))
                # print("obs: " + str(next_obs))
                # print(next_obs.size)
                # print("info" + info)
                
                # Update the observation: add the next_obs to the frame stack 
                experience_buffer.add_frame(next_obs)

                # Now save the experience to the memory
                mc_agent.replay_buffer.add_memory((state, action, reward, experience_buffer.get_stacked_frames(), done))
                
                # Test: Save images
                if saveimagesteps > 0 and steps % saveimagesteps == 0:
                    img = Image.fromarray(next_obs.reshape(h, w, c))
                    img.save(f'{image_dir}/image_{episode}_{steps}.png')

                # Train the network
                mc_agent.train()

                # Update tqdm bar
                step_bar.update(1)

                # time.sleep(.05) # Wait to get server response (may be not necessary because mc_agent.train() takes time)
        
        # Decrease epsilon after each episode
        if mc_agent.epsilon > mc_agent.epsilon_end:
            mc_agent.epsilon *= mc_agent.epsilon_decay

        if done == True and reward >= 0.98: # Has to be the reward defined in the XML (current reward for (reaching goal = 1) + (-0.01 per Step))
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
            mc_agent.create_checkpoint(checkpoint_path, memories_path, episode, completions, base_name)
        
        # Create permanent checkpoints for evaluation
        # if episode % permanent_checkpoint_interval == 0:
        #     permanent_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth')
        #     permanent_memories_path = os.path.join(checkpoint_dir, f'memories_ep{episode}.pkl')
        #     mc_agent.create_checkpoint(permanent_checkpoint_path, permanent_memories_path, episode, completions, base_name)
            
    writer.close()
    env.close()


def evaluate():
    """
    Evaluate the performance of the trained Minecraft agent.
    Loads a trained model, runs the agent in the evaluation environment and tracks its performance.
    Downloads the model if it is missing.
    The evaluation can involve multiple agents running simultaneously.
    ---------

    Returns:
        None.
    """
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
    """
    Run the evaluation for a single agent role.
    This function handles the agent's actions during evaluation, updating its state, performing actions and logging rewards.
    It runs until the agent either succeeds, fails or the global stop event is triggered.
    ---------

    Args:
        role (int): The role of the agent (0 for agent, 1 for spectator).
        global_stop_event (Event): Event to signal when to stop the evaluation.
        agent_done_event (Event): Event to signal when the agent is done.
        xml (str): The XML configuration for the mission.
        mc_agent (Agent): The trained agent that will perform the evaluation.
    ---------

    Returns:
        None.
    """
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
        h, w, c = env.observation_space.shape

    # Main evaluate loop
    while not global_stop_event.is_set():
        # Initial observation and creation ExperienceBuffer
        obs = env.reset()
        if role == 0:
            experience_buffer = ExperienceBuffer(obs, h, w, c) # Will be recreated every episode
        steps = 0
        done = False
        total_reward = 0
        
        time.sleep(1)
        # One episode
        while not done and not global_stop_event.is_set() and (episodemaxsteps <= 0 or steps < episodemaxsteps):
            steps += 1

            if role == 0: # Agent
                state = experience_buffer.get_stacked_frames()

                # Select and perform action (random or via model)
                action = mc_agent.select_action(state)
                next_obs, reward, done, info = env.step(action)

                # Check if next_obs is valid (not empty or invalid)
                if next_obs is None or next_obs.size == 0:
                    print(f"Warning: Encountered empty observation at step {steps}. Ending episode early.")
                    done = True
                    agent_done_event.set()
                    break
                
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

            if steps >= episodemaxsteps and not done:
                print("Agent has reached max steps.")
            elif done and role == 0:
                agent_done_event.set()
                print("Agent is done.")

            time.sleep(1) # Decide how fast the agent moves

        if role == 0:
            agent_done_event.clear()

    env.close()

def wait_for_stop(global_stop_event):
    """
    Waits for the user to press Enter to stop the evaluation loop.
    ---------

    Args:
        global_stop_event (Event): Event to trigger when the user chooses to stop.
    ---------

    Returns:
        None.
    """
    input("Press Enter to stop...\n")
    global_stop_event.set()

def log(message):
    """
    Formats and prints a message to the console with the role of the agent included.
    ---------
    
    Args:
        message (str): The message to log.
    ---------

    Returns:
        None.
    """
    print(f'[{role}] {message}')

def start_tensorboard(log_dir=None, download=False):
    """
    Start TensorBoard for visualizing training logs.
    --------

    Args:
    log_dir (str):
        The directory containing TensorBoard logs. Defaults to None.
    download (bool):
        If True, download logs from Google Drive before starting TensorBoard. Defaults to False.
    ---------

    Returns:
        None.
    """
    if not os.path.exists('runs'):
        os.makedirs('runs')
    
    if download:
        log_url = "https://drive.google.com/file/d/1BimbzsvLJVmOSMbnMvYY_zb0Uo4YLOKj/view?usp=sharing"

        try:
            print("Downloading logs from Google Drive.")
            output_path = os.path.join('runs', 'logs.zip')
            gdown.download(log_url, output_path, fuzzy=True, use_cookies=False, quiet=False)
            print("Logs downloaded successfully.")

            # Unzip
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                extracted_folder = zip_ref.namelist()[0]
                extraction_path = os.path.join('runs', extracted_folder)
                zip_ref.extractall('runs')
                print(f"Logs extracted to {extraction_path}.")
            
            # Remove ZIP
            os.remove(output_path)

            log_dir = extraction_path

        except Exception as e:
            print(f"Failed to download logs: {e}")
            return
        
    if log_dir is None:
        dirs = [d for d in os.listdir('runs') if d.startswith('training_')]

        if not dirs:
            print("No log directories found.")
            return
        else:
            dirs.sort(reverse=True)  # Use the latest training directory
            log_dir = os.path.join('runs', dirs[0])

    # Start TensorBoard
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])
        print(f"TensorBoard is running on http://localhost:6006 with logs from {log_dir}")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

if __name__ == '__main__':

    checkpoint_dir = 'checkpoints'
    trace_length = 4 # Images for experience buffer

    # Training parameters
    episodes = 100000
    episodemaxsteps = 200 # Change according to map
    replay_size = 150000 # Memory amount (number of memories (steps, each: current 4 frames & latest 3 + new frame)) for replay buffer (needs to be adjusted to fit RAM-size)
    batch_size = 128 # Amount of memories to be used per training-step
    saveimagesteps = 1 # Training: 0 = no images will be saved, e.g. 2 = every 2 steps an image will be saved
    resume_episode = 0
    completions = 0
    checkpoint_interval = 1000
    permanent_checkpoint_interval = 10000

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    memories_path = os.path.join(checkpoint_dir, 'memories.pkl')

    # Check what the user want to do
    parser = argparse.ArgumentParser(description="Train or evaluate the Minecraft agent and visualize the training progress with TensorBoard.")
    parser.add_argument('--train', action='store_true', help='Train the Minecraft agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the Minecraft agent')
    parser.add_argument('--tensorboard', action='store_true', help='Start TensorBoard')
    parser.add_argument('--download', action='store_true', help='Download TensorBoard logs from Google Drive')
    parser.add_argument('--logdir', type=str, default=None, help='Directory for TensorBoard logs')
    args = parser.parse_args()

    mission = 'maze_mission_single_agent.xml'
    mission_eval = 'maze_mission_multi_agent.xml'
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
        if args.tensorboard:
            if args.download:
                start_tensorboard(args.logdir, download=True)
            else:
                start_tensorboard(args.logdir)
        elif args.train:
            train()
        elif args.eval:
            evaluate()
        else:
            print("Choose between --train or --eval or --tensorboard")
    except KeyboardInterrupt:
        print("\nAborted by user.")
