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

MANHATTAN_CACHE = np.zeros(10000, dtype=int)
for s in range(10000):
    ar = s // 1000
    ac = (s // 100) % 10
    cr = (s // 10) % 10
    cc = s % 10
    MANHATTAN_CACHE[s] = abs(ar - cr) + abs(ac - cc)

def manhattan(state):
    return MANHATTAN_CACHE[state]


def get_reward(prev_state, next_state, action, done, runtime):
    # Terminal rewards
    if done and not runtime:
        return 100.0
    elif runtime:
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

    alpha = 0.15          # learning rate
    gamma = 0.98         # discount factor
    epsilon = 1.0        # exploration rate
    epsilon_min = 0.05   # lowest epsilon allowed
    epsilon_max = 1
    epsilon_decay = 0.002  # fast decay

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
               
        state = env.reset()
        done = False
        if isinstance(state, tuple):
                state = state[0]

        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, _, done, runtime, info = env.step(action)

            if isinstance(state, tuple):
                state = state[0]

            # FIXED — correct shaped reward call
            reward = get_reward(state, next_state, action, done, runtime)

            # Q-learning update
            best_next = np.max(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])

            state = next_state

        epsilon = max(epsilon_min, epsilon * np.exp(-epsilon_decay))
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table