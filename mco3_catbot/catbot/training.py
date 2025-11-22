import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

def manhattan(state):
    ar = state // 1000
    ac = (state // 100) % 10
    cr = (state // 10) % 10
    cc = state % 10
    return abs(ar - cr) + abs(ac - cc)

def get_reward(prev_state, next_state, action, done, info):
    # Terminal rewards
    if done and info.get("caught", False):
        return 100.0
    elif done and info.get("timeout", False):
        return -75.0
    
    reward = 0.0

    # Small step penalty
    reward -= 0.05

    # Distance shaping
    prev_d = manhattan(prev_state)
    next_d = manhattan(next_state)

    if next_d < prev_d:
        reward += 2.5
    elif next_d == prev_d:
        reward -= 0.2
    else:
        reward -= 2.0

    # Wall penalty (tried to move but nothing changed)
    if next_state == prev_state and action != 4:
        reward -= 3.0

    return reward   

#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    alpha = 0.1          # learning rate
    gamma = 0.9          # discount factor
    epsilon = 1.0        # exploration rate
    epsilon_min = 0.05   # lowest epsilon allowed
    epsilon_decay = 0.999  # slow decay

    # For shaping without adding functions outside allowed area:
    # We store previous Manhattan distance temporarily during training.
    prev_distance = 0

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        state, info = env.reset()
        done = False

        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, _, done, _, info = env.step(action)

            # FIXED — correct shaped reward call
            reward = get_reward(state, next_state, action, done, info)

            # Q-learning update
            best_next = np.max(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table