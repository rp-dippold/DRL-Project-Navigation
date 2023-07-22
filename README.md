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
 ┣ 📂reward_plots **`(contains the reward plots to successfuly trained agents)`** <br>
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

The code of this project was tested on Linux and Windows 11. To get the code running on your local system, follow these steps
which are base on Anaconda and pip:

1.  `conda create --name banana python=3.8 -c conda-forge`
2.  `conda activate banana`
3.  `[ONLY WINDOWS] conda install swig -c conda-forge`
4.  Create a directory where you want this save this project
5.  `git clone https://github.com/rp-dippold/DRL-Project-Navigation.git`
6.  `cd python`
7.  `pip install .`
8.  `python -m ipykernel install --user --name banana --display-name "banana"`
9.  Install Pytorch:
    * [CPU]: `pip install torch torchvision torchaudio`
    * [GPU]: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`.\
    Depending on your GPU and cudnn version a different pytorch version my be required. Please refer to 
    https://pytorch.org/get-started/locally/.


## Instructions
The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here.