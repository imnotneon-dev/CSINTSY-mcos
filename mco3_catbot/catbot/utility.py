import time
import pygame
import numpy as np
from cat_env import make_env

def play_game(env):
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        env.render()
        
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # up
                elif event.key == pygame.K_DOWN:
                    action = 1  # down
                elif event.key == pygame.K_LEFT:
                    action = 2  # left
                elif event.key == pygame.K_RIGHT:
                    action = 3  # right
                elif event.key == pygame.K_q:
                    env.close()
                    return
        
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if done:
                print(f"Game Over! Total reward: {total_reward:.2f}")
                env.render()
                time.sleep(2)
                
    env.close()

def play_q_table(env, q_table, move_delay=0.25, max_steps=1000, window_title=None):
    obs, _ = env.reset()
    env.render()
    if window_title:
        pygame.display.set_caption(window_title)

    done = False
    total_reward = 0.0
    next_move_time = time.time() + move_delay
    moves = 0

    while not done:
        env.render()
        current_time = time.time()
        if current_time >= next_move_time:
            action = int(np.argmax(q_table[obs]))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            next_move_time = current_time + move_delay
            moves += 1
            done = terminated or truncated
            if moves >= max_steps:
                done = True
                break
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    env.close()
                    return

    env.render()
    time.sleep(1)
    env.close()
    return terminated