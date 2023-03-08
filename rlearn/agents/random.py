from rlearn.agents import Agent
import gym

class RandomAgent(Agent):
    '''A random agent
    '''
    @classmethod
    def get_supported_action_space_types(cls):
        return ["discrete"]

    def __init__(self, env : gym.Env, config : dict):
        super().__init__(env = env, config = config)
    
    def act(self, obs):
        action = self.env.action_space.sample()
        self.compute_metrics(mode = 'act')
        return action
    
    def learn(self):
        self.compute_metrics(mode = 'learn')
        pass
    
    def remember(self, *args):
        self.compute_metrics(mode = 'remember')
        pass