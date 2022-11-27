from rlearn.memory import Memory_episodic
from rlearn.metrics import MetricS_On_Learn, MetricS_On_Learn_Numerical, Metric_Performances
from rlearn.algorithms import Agent
import gym

class RandomAgent(Agent):
    '''A random agent
    '''
    space_types = ['discrete', 'continuous', 'obs-continuous', 'action-continuous']

    def __init__(self, env : gym.Env, agent_cfg : dict):
        super().__init__(
            env = env, 
            agent_cfg = agent_cfg,
            metrics=[MetricS_On_Learn_Numerical, Metric_Performances]) #Choose metrics here
    
    def act(self, obs):
        action = self.env.action_space.sample()
        self.add_metric('act')
        return action
    
    def learn(self):
        self.add_metric('learn')
        pass
    
    def remember(self, *args):
        self.add_metric('remember')
        pass