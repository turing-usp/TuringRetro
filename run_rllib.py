import argparse
import numpy as np

import retro
from utils import retro_wrappers

import ray
from ray.rllib.agents import ppo, impala, dqn
from utils.rllib_utils import register_retro, train, test

parser = argparse.ArgumentParser()
parser.add_argument("game", type=str)
parser.add_argument("state", type=str)
parser.add_argument("-c", "--checkpoint", type=str)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-a", "--agent", type=str, default="PPO")
parser.add_argument("-f", "--framework", type=str, default="tf")
parser.add_argument("-e", "--episodes", type=int, default=1)

args = parser.parse_args()

if __name__ == "__main__":
    game = args.game
    state = args.state
    wrapper = retro_wrappers.get_wrapper(game)
    checkpoint = args.checkpoint
    training = args.train
    agent = args.agent
    framework = args.framework
    episode_count = args.episodes

    info = ray.init(ignore_reinit_error=True)
    
    register_retro(game, state, wrapper)
    
    if agent == "PPO":
        trainer_config = ppo.DEFAULT_CONFIG.copy()
        trainer_config['num_workers'] = 0
        trainer_config["num_cpus_per_worker"] = 4
        trainer_config["num_envs_per_worker"] = 1
        trainer_config['lambda'] = 0.95
        trainer_config['kl_coeff'] = 0.5
        trainer_config['clip_param'] = 0.1
        trainer_config['vf_clip_param'] = 10.0
        trainer_config['entropy_coeff'] = 0.01
        trainer_config["train_batch_size"] = 500
        trainer_config['rollout_fragment_length'] = 100
        trainer_config['sgd_minibatch_size'] = 128
        trainer_config['num_sgd_iter'] = 10
        trainer_config['batch_mode'] = "truncate_episodes"
        trainer_config['observation_filter'] = "NoFilter"
        trainer_config['framework'] = 'tf' if framework == "tf" else 'torch'
            
        agent = ppo.PPOTrainer(config=trainer_config, env=game)
    elif agent == "APEXDQN":
        trainer_config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
        trainer_config['log_level'] = "WARN"
        trainer_config['clip_rewards'] = True
        trainer_config["num_gpus"] = 1
        trainer_config['output'] = './checkpoints/' 
        trainer_config['target_network_update_freq'] = 20000
        trainer_config["remote_worker_envs"] = True
        trainer_config["num_workers"] = 4
        trainer_config['num_envs_per_worker'] = 2
        trainer_config['lr'] = .00005
        trainer_config['train_batch_size'] = 64
        trainer_config['gamma'] = 0.99
        agent = dqn.apex.ApexTrainer(config=trainer_config, env=game)
    elif agent == "IMPALA":
        trainer_config = impala.DEFAULT_CONFIG.copy()
        trainer_config['log_level'] = "WARN"
        trainer_config['clip_rewards'] = True
        trainer_config["num_gpus"] = 1
        trainer_config['output'] = './checkpoints/' 
        trainer_config['rollout_fragment_length'] = 50
        trainer_config['train_batch_size'] = 500
        trainer_config["remote_worker_envs"] = True
        trainer_config['num_workers'] = 8
        trainer_config['num_envs_per_worker'] = 4
        trainer_config['lr_schedule'] = [           
                [0, 0.0005],
                [20000000, 0.000000000001],
        ]
        trainer_config['framework'] = 'tf' if framework == "tf" else 'torch'
        
        agent = impala.ImpalaTrainer(config=trainer_config, env=game) 

    if training:
        trainer = train(agent, checkpoint=checkpoint)
    else:
        test(agent, game, state, wrapper, checkpoint=checkpoint, render=True, episode_count=episode_count)