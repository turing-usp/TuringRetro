import warnings
import retro
from gym.wrappers import Monitor
from ray.tune import register_env

def retro_env_creator(game, state, scenario, wrapper):
    base = retro.make(game=game, state=state, scenario=scenario)
    base = wrapper(base, transpose=False)
    return base

def register_retro(game, state, scenario, wrapper):
    env_creator = lambda env_config: retro_env_creator(game, state, scenario, wrapper)
    register_env(game, env_creator)

def train(agent, checkpoint=None, iterations=1000000, save_every=25, save_path="./checkpoints/"):
    if checkpoint is not None:
        try:
            agent.restore(checkpoint)
            print(f"-------------------------------\n"
                  f"Resumed checkpoint {checkpoint}\n"
                  f"-------------------------------\n")
        except:
            print(f"------------------------\n"
                  f"Checkpoint not found: restarted policy network from scratch\n"
                  f"------------------------\n")

    s = "Iteração: {:3d}, Recompensas (Min/Mean/Max): {:6.2f}/{:6.2f}/{:6.2f}, Duração Média: {:6.2f}"

    for i in range(iterations):
        result = agent.train()
        
        if i % save_every == 0:
            checkpoint = agent.save(save_path)
            print("Checkpoint salvo em: ", checkpoint)
            
        print(s.format(
            i + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
        ))

def test(agent, game, state, scenario, wrapper, checkpoint=None, render=False, record=False, episode_count=1, maxepisodelen=10000):
    """Tests and renders a previously trained model"""
    if checkpoint is None:
        warnings.warn("Running without a previously trained checkpoint")
    else:
        agent.restore(checkpoint)
    
    agent.cleanup()

    env = retro_env_creator(game, state, scenario, wrapper)
    
    if record:
        env = Monitor(env, './videos/', force=True)

    for _ in range(episode_count):
        state = env.reset()
        done = False
        reward_total = 0.0
        step = 0
        while not done and step < maxepisodelen:
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            if render:
                env.render()
            state = next_state
            step = step + 1

    env.close()
