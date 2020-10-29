import gym
import numpy as np

class Callback(object):
    def __init__(self, frequency=1):
        self.frequency = frequency
        self.counter = 0
    
    def update(self, agent):
        self.counter = (self.counter + 1) % self.frequency
        if self.counter == 0:
            self.run(agent)

    def run(self, agent):
        raise NotImplementedError
class CallbackList(Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def update(self, agent):
        for callback in self.callbacks:
            callback.update(agent)

class EvalCallback(Callback):
    def __init__(self, env, episode_count=1, frequency=1):
        super(EvalCallback, self).__init__(frequency)
        self.env = env
        self.episode_count = episode_count
        self.max_score = -np.inf

    def update(self, agent):
        super(EvalCallback, self).update(agent)

    def run(self, agent):
        episode_returns = []

        timestep = 0
        episode = 0

        for episode in range(self.episode_count):
            state = self.env.reset()
            total_reward = 0
            done = 0
            while not done:
                action = agent.act(state, greedy=True)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            
            episode_returns.append(total_reward)
        
        mean_score = np.mean(episode_returns)
        
        print(f"\n Evaluation: Mean Score = {mean_score:5.3f}, Max Score = {self.max_score:5.3f}")

        if mean_score >= self.max_score:
            self.max_score = mean_score
            agent.save_model("saved_models/best_model.pth")

class EpsilonCallback(Callback):
    def __init__(self, epsilon_reset=0.5, frequency=1):
        super(EpsilonCallback, self).__init__(frequency)
        self.epsilon_reset = epsilon_reset

    def update(self, agent):
        super(EpsilonCallback, self).update(agent)

    def run(self, agent):
        agent.epsilon = self.epsilon_reset

class SaveCallback(Callback):
    def __init__(self, frequency=1):
        super(SaveCallback, self).__init__(frequency)

    def update(self, agent):
        super(SaveCallback, self).update(agent)

    def run(self, agent):
        agent.save_model("saved_models/latest_model.pth")