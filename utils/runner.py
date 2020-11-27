import math
import time
import numpy as np
from collections import deque
from gym.wrappers import Monitor

def train(agent, env, total_timesteps, callback):
    total_reward = 0
    episode_returns = deque(maxlen=20)
    avg_returns = []

    state = env.reset()
    start_time = time.time()
    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.train()
        timestep += 1

        total_reward += reward

        if done:
            episode_returns.append(total_reward)
            episode += 1
            callback.update(agent)
            next_state = env.reset()

        if episode_returns:
            avg_returns.append(np.mean(episode_returns))

        total_reward *= 1 - done
        state = next_state

        ratio = math.ceil(100 * timestep / total_timesteps)
        uptime = math.ceil(time.time() - start_time)

        avg_return = avg_returns[-1] if avg_returns else np.nan
        
        print(f"\r[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d}, avg_return = {avg_return:10.4f}", end="")

    return avg_returns

def test(agent, env, episodes, record=False):
    if record:
        env = Monitor(env, './videos/', force=True)

    for episode in range(episodes):
        done = 0
        state = env.reset()
        total_reward = 0

        while not done:
            action = agent.act(state, greedy=True)
            state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward

            print(f"\r {total_reward:3.3f}", end="")
        
    env.close()
