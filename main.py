import math
import retro
import numpy as np
from collections import deque

from agents.dqn.agent import DQNAgent
from utils import retro_wrappers
from utils.callbacks import EvalCallback, EpsilonCallback, CallbackList, SaveCallback

def main():
    game_rom = "MegaMan2-Nes" #Nome da rom
    state = "Normal.Metalman.Fight.state" 
    scenario = "scenario"
    env = retro.make(game_rom, state=state, scenario=scenario)
    env = retro_wrappers.wrap_retro(env)

    eval_callback = EvalCallback(env, frequency=10, episode_count=1)
    epsilon_callback = EpsilonCallback(frequency=100)
    saving_callback = SaveCallback(frequency=1)

    callbacks = CallbackList([eval_callback, epsilon_callback, saving_callback])

    OBS_SPACE = env.observation_space
    ACT_SPACE = env.action_space
    BATCH_SIZE = 32
    MAX_MEMORY = 10000
    N_STEP = 3
    ALPHA = 0.7
    BETA = 0.5
    BETA_DECAY = 1e-5
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    TAU = 0.01
    EPS_INIT = 0.9
    EPS_END = 0.025
    EPS_DECAY = 0.9995

    agent = DQNAgent(observation_space=OBS_SPACE, 
                 action_space=ACT_SPACE,
                 batch_size=BATCH_SIZE,
                 max_memory=MAX_MEMORY,
                 n_step=N_STEP,
                 alpha=ALPHA,
                 beta=BETA,
                 beta_decay=BETA_DECAY, 
                 lr=LEARNING_RATE, 
                 gamma=GAMMA,
                 tau=TAU, 
                 epsilon_init=EPS_INIT,
                 epsilon_decay=EPS_DECAY,
                 min_epsilon=EPS_END)
    
    returns = train(agent, env, 1000000, callbacks)

def train(agent, env, total_timesteps, callback):
    total_reward = 0
    episode_returns = deque(maxlen=20)
    avg_returns = []

    state = env.reset()
    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.train()
        # env.render()
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

        avg_return = avg_returns[-1] if avg_returns else np.nan
        
        print(f"\r[{ratio:3d}%] timestep = {timestep}/{total_timesteps}, episode = {episode:3d}, avg_return = {avg_return:10.4f}, epsilon [{100*agent.epsilon:.2f}%],  loss = [{loss:2f}]", end="")

    return avg_returns

if __name__ == "__main__":
    main()