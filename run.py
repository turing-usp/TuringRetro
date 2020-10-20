import math
import retro
from collections import deque

from agent import *
from utils import retro_wrappers

def run():
    game_rom = "SuperMarioKart-Snes"
    env = retro.make(game_rom)
    env = retro_wrappers.wrap_mario_kart(env)

    BATCH_SIZE = 32
    ALPHA = 0.7
    BETA = 0.5
    BETA_DECAY = 1e-5
    GAMMA = 0.99
    EPS_INIT = 0.9
    EPS_END = 0.025
    EPS_DECAY = 0.9995
    TAU = 0.01
    MAX_MEMORY = 10000
    OBS_SPACE = env.observation_space
    ACT_SPACE = env.action_space
    N_STEP = 3

    agent = DQNAgent(observation_space=OBS_SPACE, 
                 action_space=ACT_SPACE,
                 alpha=ALPHA,
                 beta=BETA,
                 beta_decay=BETA_DECAY, 
                 lr=7e-4, 
                 gamma=GAMMA,
                 tau=TAU, 
                 max_memory=MAX_MEMORY,
                 epsilon_init=EPS_INIT,
                 epsilon_decay=EPS_DECAY,
                 min_epsilon=EPS_END,
                 n_step=N_STEP)

    agent.load_model("best_model.pth")
    
    test(agent, env, 1)

def test(agent, env, episodes):
    for episode in range(episodes):
        done = 0
        state = env.reset()
        total_reward = 0

        while not done:
            action = agent.act(state, greedy=True)
            state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
        
        print(total_reward)
    env.close()

if __name__ == "__main__":
    run()