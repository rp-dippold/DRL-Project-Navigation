import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from collections import deque
from unity_agent import Agent
from unityagents import UnityEnvironment


def train_agent(watch=False):
    """Train an agent for an environment.

    If argument `watch` is set to 'True', the user can watch the agent during
    training. Default is 'False'

    Parameters
    ----------
    watch : bool (Default False)
        Shows environment during training if set to 'True'.

    Returns
    -------
        A list of the scores of all episodes collected during training.
        If the environment was solved by the agent, its parameters
        (state dictionar) is saved.
    """
    config = Config.get_config()
    
    # Collection information regarding the environment and the agent (brain)
    env = UnityEnvironment(file_name=config.banana_env,
                           no_graphics=not watch)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    
    # Initialize agent
    agent = Agent(state_size=state_size, 
                  action_size=action_size,
                  hidden_sizes=config.hidden_sizes)
    
    # List containing scores from each episode
    scores = [] 
    # Queue containing the amount of scores specified by scores_window
    scores_window = deque(maxlen=config.scores_window)
    eps = config.epsilon_start  # initialize epsilon
    beta = config.beta_start    # initialize beta
    for i_episode in range(1, config.n_episodes+1):
        # Reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # Obtain state of the initial environment
        state = env_info.vector_observations[0]
        score = 0
        # Play the game until it terminates (done) or at most max timesteps
        for t in range(config.max_timesteps):
            # Get the action for the current state from the agent
            action = agent.act(state, eps)
            # Take the next step using the received action
            env_info = env.step(action)[brain_name]
            # Extract the next state from the environment
            next_state = env_info.vector_observations[0]
            # Get the reward for the last action
            reward = env_info.rewards[0]
            # Find out if the game is over (True) or still running (False)
            done = env_info.local_done[0]
            # Inform the agent about the result of the last step and
            # allow the agent to learn from it.
            agent.step(
                state, action, reward, next_state, done, config.alpha, beta)
            # Roll over the state to next time step and update score
            state = next_state
            score += reward
            # Exit loop if episode has finished
            if done:
                break 

        # Save most recent score
        scores_window.append(score)
        scores.append(score)
        # Update epsilon and beta
        eps = max(config.epsilon_end, config.epsilon_decay*eps)
        beta = min(beta/config.beta_increase, config.beta_end)
        
        # Print result of the episode on the terminal screen
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= config.total_avg_reward:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: \
                  {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # Save agent parameters
            hidden_s = f'{config.hidden_sizes}'.replace(', ', '-')[1:-1]
            rslt = f'{hidden_s}-{np.mean(scores_window):.2f}-{i_episode-100}'
            filename = f'checkpoint-{rslt}.pth'
            torch.save(agent.dqnetwork_local.state_dict(), filename)
            break
    env.close()
    return scores

def run_model(params_file):
    config = Config.get_config()
    env = UnityEnvironment(file_name=config.banana_env)

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # Number of states and actions
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size

    # Create default (untrained) agent
    agent = Agent(state_size=state_size, 
                  action_size=action_size,
                  hidden_sizes=config.hidden_sizes)
    # Load trained parameters
    agent.dqnetwork_local.load_state_dict(torch.load(params_file))

    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name] # Reset the environment
    state = env_info.vector_observations[0]            # Get the current state
    score = 0                                          # Initialize the score
    while True:
        action = agent.act(state)                      # Select an action
        env_info = env.step(action)[brain_name]        # Send the action to 
                                                       # environment
        next_state = env_info.vector_observations[0]   # Get the next state
        reward = env_info.rewards[0]                   # Get the reward
        done = env_info.local_done[0]                  # See if episode has 
                                                       # finished
        score += reward                                # Update the score
        state = next_state                             # Roll over the state to
                                                       # next time step
        if done:                                       # Exit loop if episode
           break                                       # finished
    
    # Print final score
    print("Score: {}".format(score))
    # Close environment after three seconds
    time.sleep(3)
    env.close()


if __name__ == '__main__':
    # Process user arguments to start the requested function
    parser = argparse.ArgumentParser(
        description="Train or run an agent to navigate in a large square \
                     world collecting yellow bananas.")
    parser.add_argument('action', type=str,
                        help="Enter 'train' for agent training and 'run' for \
                              running a trained agent.")
    parser.add_argument('--params', type=str,
                        help="Filepath of the parameters to be used in the \
                              agent's model to run the game.")
    parser.add_argument('--watch', action='store_true',
                        help='Watch agent during training.')

    args = parser.parse_args()
    
    # Run the game with a smart agent
    if args.action == 'run':
        if args.params is None:
            raise ValueError('Cannot initialize agent due to missing \
                              argument for model parameters!')
        else:
            run_model(args.params)
    # Train an agent
    elif args.action == 'train':
        scores = train_agent(args.watch)
        # Plot rewards
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        print('Unkown action! Possible actions are "run" and "train".')