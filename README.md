# Deep Reinforcement Learning - Project-Navigation

## Project Details
This project is based on a Unity environment to design, train, and evaluate deep reinforcement learning algorithms.
The environment used in this project is a large square world which contains blue and yellow bananas.

An agent is trained to navigate in this world and to collect as much yellow bananas as possible and to avoid blue bananas.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes. To make it a little bit more challenging, the average score to be attained was set to +16.

## Getting Started

### Project File Structure
The project is structured as follows:

📦project<br>
 ┣ 📂Banana_Linux  **`(contains the Banana environment for Linux based systems)`** <br>
 ┣ 📂Banana_Windows_x86_64  **`(contains the Banana environment for Windows 64-bit based systems)`** <br>
 ┣ 📂models  **`(contains the model states of successfully trained agents)`** <br>
 ┃ ┣ checkpoint-16-16-16.00-625.pth<br>
 ┃ ┗ ...<br>
 ┣ 📂python **`(files required to set up the environment)`** <br>
 ┣ 📂reward_plots **`(contains the reward plots of successfully trained agents)`** <br>
 ┃ ┣ Reward-Plot-16-16-16_00-625.jpeg<br>
 ┃ ┗ ...<br>
 ┣ .gitignore <br>
 ┣ config.py  <br>
 ┣ config.yml <br>
 ┣ main.py **`(Python script to run a trained agent or to train a new one)`**<br>
 ┣ model.py **`(DQ-Network)`**<br>
 ┣ README.md <br>
 ┣ Report.md <br>
 ┗ unity_agent.py **`(Unity agent for banana environment)`**<br>
 
### Installation and Dependencies

The code of this project was tested on Linux (Ubuntu 20.04) and Windows 11. To get the code running on your local system, follow these steps
which are base on Anaconda and pip:

1.  `conda create --name banana python=3.8 -c conda-forge`
2.  `conda activate banana`
3.  Create a directory where you want this save this project
4.  `git clone https://github.com/rp-dippold/DRL-Project-Navigation.git`
5.  `cd python`
6.  `pip install .`
7.  `python -m ipykernel install --user --name banana --display-name "banana"`
8.  Install Pytorch:
    * [CPU]: `pip install torch torchvision torchaudio`
    * [GPU]: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`.\
    Depending on your GPU and cudnn version a different pytorch version my be required. Please refer to 
    https://pytorch.org/get-started/locally/.


## Instructions
To run the code go into the directory where you install the repository. First of all, open the file `config.yml` and check if `banana_env` refers to the correct banana environment. Set the banana environment as follows:

* **`Windows 11`**: "./Banana_Windows_x86_64/Banana.exe"
* **`Linux`**: "./Banana_Linux/Banana.x86_64"

#### Training an Agent
Before training an agent you should adapt the respective hyperparameters in config.yml. The current values allow to train an agent that can get an average score of +16 over 100 consecutive episodes.

To start training just enter the following command: `python main.py train`

If you want to watch the agent during training enter: `python main.py train --watch`

At the end of the training a window pops up that shows the scores of the agent for each episode. After closing this 
window, the program stops.

If the agent was trained successfully its weights are saved in the root directory and not in directory `models`. \
The filename matches the following pattern: `checkpoint-<hl1>-...-<hln>-<score>-<episode>.pth`
* `<hlx>` means the number of neurons in hidden layer x
* `<score>` means the average score over the last 100 consecutive episodes
* `<episode>` means the episode in which the agent solved the environment successfully (see project details.)

#### Running the Environment with a Smart Agent
Running the environment with a trained agent requires the parameters in config.yml fit the saved agent weights, i.e.
`hidden_sizes` must equal the values during training. The following command runs the environment:

`python main.py run --params <path to stored weights>`

`<path to stored weights>` is the path to the directory plus the name of the `checkpoint-xxx.path` file, e.g.
`./models/checkpoint-16-16-16.00-625.pth`.

