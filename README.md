# Minecraft Reinforcement Learning

This project implements **Double Deep Q-Learning (DQN)** with a replay memory and a neural network using **PyTorch**. 
The agent learns to navigate through a maze to a specified goal using visual input within a RL environment.
[**Minecraft Malmo**](https://github.com/microsoft/malmo) is used for the implementation, providing a platform to utilize Minecraft as a training environment for Reinforcement Learning. 
**TensorBoard** is employed to visualize the learning progress, tracking and displaying the agent's performance throughout the training process.

<!-- GIF -->
The goal of the agent is to find an optimal strategy to navigate through the maze without falling into lava, while passing through intermediate waypoints (sandstone) and completing the task by stepping on the diamond block in as few steps as possible.

---

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model](#model)
6. [Next steps](#next-steps)

---

## Features
- **Train an Agent in Minecraft**: Train an AI agent to navigate through a maze in Minecraft.
- **Real-Time training monitoring with TensorBoard**: Visualize training progress and key metrics (like rewards, losses and steps).
- **Evaluate trained agent**: Test a trained agent's performance, allowing you to see how well the agent performs.
- **Checkpoints**: Save and load training checkpoints, so you can pause and resume training at any point.
- **Save Agent's frames**: Optionally save each frame of the agent's actions during training, allowing you to create a video to showcase the agent's learning progress.
- **Create custom missions**: Customize mazes through the mission XML configuration.
---

## Prerequisites
- Python 3.11 or higher
- Java 8 JDK
- Git 2.42.0 or higher

It is essential to have the correct version of Java installed: **Java 8 JDK** ([AdoptOpenJDK](https://adoptopenjdk.net/)).  
Make sure Java is usable in the terminal by running `java --version`.  
For Windows systems, make sure to properly set the **Path variable** and for Linux, ensure that **JAVA_HOME** is configured correctly.  
If necessary, uninstall any previous Java versions to ensure everything works as expected. 

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tis22/Minecraft_RL.git
    cd Minecraft_RL
    ```
   
2. Create a virtual environment and install the required dependencies from the `requirements.txt` file:
     
    ```bash
    python -m venv minecraft_rl
    # On Windows: .\minecraft_rl\Scripts\activate
    # On Linux: source minecraft_rl/bin/activate
    pip install -r requirements.txt
    ```

### Installing the Minecraft Mod

Once the virtual environment is set up and the dependencies are installed, make sure you are in your user directory.
- **Windows**: `C:\Users\username`
- **Linux**: `/home/username` or `~/` (for the home directory)

You can complete the installation of the Minecraft Mod by running the following command:

```bash
python -c "import malmoenv.bootstrap; malmoenv.bootstrap.download()"
```

If you encounter any issues during installation of Minecraft MalmoEnv,  
please refer to the official [installation guide](https://github.com/microsoft/malmo/tree/master/MalmoEnv) in the corresponding repository for more detailed instructions.

### Copying Repository Files to MalmoEnv
After the Minecraft Mod has been successfully installed, a folder named **MalmoPlatform** should appear in the user directory:

- **Windows**: `C:\Users\username\MalmoPlatform`
- **Linux**: `/home/username/MalmoPlatform` or `~/MalmoPlatform` (for the home directory)

Navigate to the **MalmoEnv** subfolder within **MalmoPlatform**.  
In this **MalmoEnv** folder, you should now copy the contents of the previously cloned repository (Minecraft_RL).

If you don't see the **MalmoPlatform** folder or any other files, make sure to enable the display of hidden files:

- **Windows**: In File Explorer, go to the "View" tab and check the "Hidden items" box.
- **Linux**: Press `Ctrl + H` in your file manager to show hidden files.

### Resolving Minecraft Resource Download Issues
It is common to encounter errors while downloading resources during Minecraft compilation, but these typically affect sound assets.
Many of these sound assets can be manually downloaded as described below, which may speed up the process. 
However, the compilation should work without manual intervention as well.
1. Ensure that the `.gradle` folder exists in your user directory.
2. Download the required `gradle_caches_minecraft.zip` [file](https://1drv.ms/u/s!AuopXnMb-AqcgdZkjmtSVg3VQL5TEQ?e=w4M4r7).
3. Rename or remove the existing Minecraft cache (if it exists):
   ```bash
   mv ~/.gradle/caches/minecraft ~/.gradle/caches/minecraft-org
   ```
   
4. Extract the contents of the ZIP file into the `.gradle/caches` directory:
    
   ```bash
   unzip gradle_caches_minecraft.zip -d ~/.gradle/caches
   ```

---

## Usage
### Training the agent
If you want to train the agent, the program checks whether there is an existing training session (e.g. the `runs`, `checkpoints` and `images` directories should exist).  
- If the last checkpoint is found, training resumes from the next episode.  
- Otherwise, a new training session starts.

During training, data for TensorBoard is saved in the runs directory.

#### Additional Settings in the Code
- To save individual frames (e.g. for creating a video afterwards), set the saveimagesteps variable to 1.
- Parameters such as the number of episodes, maximum steps per episode, replay memory size and batch size can also be configured.
- By default, a checkpoint is created every 1000 episodes.

#### Steps to Start Training
1. Launch the environment in the first terminal:
   ```bash
   source minecraft_rl/bin/activate
   python -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"
   ```
   *This opens a new window displaying the environment, as defined in the mission XML (default: 84x84).*  
   **Wait until the Minecraft Launcher window appears before proceeding.**

3. Start training in a second terminal:
   ```bash
   source minecraft_rl/bin/activate
   cd MalmoPlatform/MalmoEnv/
   python main.py --train
   ```

To stop training, press `Ctrl + C`.


### Evaluating the trained agent
To evaluate a trained agent, use the `--evaluate` flag.  
- The program uses the trained model stored on the disk.  
- If no local model is found, it downloads the model from Google Drive.

### Steps to Start Evaluation
1. Launch Minecraft in the first terminal:
   ```bash
   source minecraft_rl/bin/activate
   python -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"
   ```
   **Wait until the Minecraft Launcher window appears before proceeding.**

2. Launch Minecraft in the second terminal:
   ```bash
   source minecraft_rl/bin/activate
   python -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9001)"
   ```
   **Wait until the Minecraft Launcher window appears before proceeding.**

3. Start evaluation in the third terminal:
   ```bash
   source minecraft_rl/bin/activate
   cd MalmoPlatform/MalmoEnv/
   python main.py --evaluate
   ```
The evaluation mode will continue running until you press `Enter` to stop it.  
The agent will repeatedly start from scratch to try and reach the goal.  
While evaluating, you will see real-time information in the terminal, such as the agent’s current step, 
the action it took, the reward it received and the total accumulated reward up to that point.



### Using TensorBoard
To visualize training progress or analyze a specific model's logs, use TensorBoard.

#### Options for TensorBoard
1. View the latest/current TensorBoard logs:
   ```bash
   python main.py --tensorboard
   ```

2. View specific TensorBoard logs:
   ```bash
   python main.py --tensorboard --logdir "path/to/logdir"
   ```

3. Download TensorBoard logs for the model from Google Drive and view:
   ```bash
   python main.py --tensorboard --download
   

---

### Model
DDQN: Convolutional Neural Network with the last 4 RGB frames (12 channels)
![CNN](https://github.com/user-attachments/assets/cd23ef9c-57c7-49bd-b3c2-35b72c82799c)


The model approximates Q-values for each possible action based on the current state (current frame & last three frames) that the agent observes and selects actions accordingly.
The actions the agent can take are: move forward, move backward, turn left and turn right, allowing it to navigate in all directions.
The agent employs epsilon-greedy exploration and learns from past experiences stored in a replay memory.

---

## Next steps
- **Mixing Experiences**: The agent will learn simultaneously on both parts of the maze. Memories from both halves will be combined to prevent overfitting to a subtask and improve generalization.
- **Adaptive Exploration**: The epsilon value will be dynamically adjusted based on the agent's previous success. If the agent stagnates or develops suboptimal strategies during a certain phase, the epsilon value could be increased again to encourage further exploration and move the agent out of local optima.
- **Prioritized Experience Replay**: Important experiences will have a higher chance of being replayed during training, strengthening the agent's ability to handle rare but critical situations.
- **Long Short-Term Memory (LSTM)**: This architecture will provide the agent with a better memory for long-term dependencies, enabling it to tackle more complex tasks effectively.
---

