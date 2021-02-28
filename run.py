import retro
import argparse

from agents.dqn.agent import DQNAgent
from agents.ppo.agent import PPOAgent
from utils.runner import test
from utils import retro_wrappers

parser = argparse.ArgumentParser()
parser.add_argument("game", type=str)
parser.add_argument("state", type=str)
parser.add_argument("-s", "--scenario", type=str, default="scenario")
parser.add_argument("-p", "--path", type=str, default="agents/saved_models/best_model.pth")
parser.add_argument("-d", "--dqn", action="store_true")
parser.add_argument("-e", "--episodes", type=int, default=1)
parser.add_argument("-r", "--render", action="store_true")

args = parser.parse_args()

def run():
    game = args.game
    state = args.state
    scenario = args.scenario
    model_path = args.path
    dqn = args.dqn
    episodes = args.episodes
    render = args.render
    wrapper = retro_wrappers.get_wrapper(game)

    env = retro.make(game)
    env = wrapper(env)

    OBS_SPACE = env.observation_space
    ACT_SPACE = env.action_space
    BATCH_SIZE = 32
    MAX_MEMORY = 10000
    N_STEP = 256
    VF_COEF = 0.5
    ENTROPY_COEF = 5e-3
    CLIP_PARAM = 0.2
    EPOCH = 10
    ALPHA = 0.7
    BETA = 0.5
    BETA_DECAY = 1e-5
    LEARNING_RATE = 7e-4
    GAMMA = 0.99
    TAU = 0.01
    LAM = 0.95
    EPS_INIT = 0.9
    EPS_END = 0.025
    EPS_DECAY = 0.9995

    if dqn:
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
    else:
        agent = PPOAgent(observation_space=OBS_SPACE,
                        action_space=ACT_SPACE,
                        lr=LEARNING_RATE,
                        gamma=GAMMA,
                        lam=LAM,
                        vf_coef=VF_COEF,
                        entropy_coef=ENTROPY_COEF,
                        clip_param=CLIP_PARAM,
                        epochs=EPOCH,
                        n_steps=N_STEP)

    agent.load_model(model_path)
    
    test(agent, env, episodes, render)

if __name__ == "__main__":
    run()