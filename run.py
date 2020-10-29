import math
import retro
from collections import deque
from gym.wrappers import Monitor

from agents.dqn.agent import DQNAgent
from utils import retro_wrappers

def run():
    game_rom = "SuperMarioKart-Snes"
    env = retro.make(game_rom)
    env = retro_wrappers.wrap_mario_kart(env)

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

    agent.load_model("saved_models/best_model.pth")
    
    test(agent, env, 1, False)

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

if __name__ == "__main__":
    run()