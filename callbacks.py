import gym
import numpy as np

class Callback(object):
    def __init__(self, frequency=1):
        self.frequency = frequency
        self.counter = 0
    
    def update(self):
        self.counter = (self.counter + 1) % self.frequency
        if self.counter == 0:
            return True
        return False

class EvalCallback(Callback):
    def __init__(self, env, episode_count=1, frequency=1):
        super(EvalCallback, self).__init__(frequency)
        self.env = env
        self.episode_count = episode_count
        self.max_score = -np.inf

    def update(self, agent):
        if super(EvalCallback, self).update():
            self.run(agent)

    def run(self, agent):
        episode_returns = []

        timestep = 0
        episode = 0

        for episode in range(self.episode_count):
            state = self.env.reset()
            total_reward = 0
            done = 0
            while not done:
                action = agent.act(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            
            episode_returns.append(total_reward)
        
        mean_score = np.mean(episode_returns)
        
        print(f"\n Evaluation: Mean Score = {mean_score:5.3f}, Max Score = {self.max_score:5.3f}")

        if mean_score >= self.max_score:
            self.max_score = mean_score
            agent.save_model("best_model.pth")
