import retro

from agents.dqn.agent import DQNAgent
from agents.ppo.agent import PPOAgent
from utils import retro_wrappers
from utils.runner import train
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

    # agent = DQNAgent(observation_space=OBS_SPACE, 
    #              action_space=ACT_SPACE,
    #              batch_size=BATCH_SIZE,
    #              max_memory=MAX_MEMORY,
    #              n_step=N_STEP,
    #              alpha=ALPHA,
    #              beta=BETA,
    #              beta_decay=BETA_DECAY, 
    #              lr=LEARNING_RATE, 
    #              gamma=GAMMA,
    #              tau=TAU, 
    #              epsilon_init=EPS_INIT,
    #              epsilon_decay=EPS_DECAY,
    #              min_epsilon=EPS_END)
    
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

    returns = train(agent, env, 1000000, callbacks)

if __name__ == "__main__":
    main()