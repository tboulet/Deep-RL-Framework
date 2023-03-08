from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from copy import deepcopy
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim

from rlearn.core.memory import Memory_episodic
from rlearn.core.metrics import ClassicalLearningMetrics
from rlearn.agents import Agent


#ACTOR PI
class ContinuousPolicy(nn.Module):
    def __init__(self, env : gym.Env):
        """Class for the policy network of the DDPG agent.
        
        Args:
            env (gym.Env): the environment
        """
        super(ContinuousPolicy, self).__init__()
        a_high = env.action_space.high
        a_low = env.action_space.low
        n_obs = env.observation_space.shape[0]
        if len(env.observation_space.shape) > 1:
            raise NotImplementedError("Only works with 1D observation spaces.")
        dim_actions = env.action_space.shape
        self.range_action = torch.Tensor(a_high - a_low)
        self.mean_action = torch.Tensor((a_high + a_low)/2)
        self.fc1 = nn.Linear(n_obs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, *dim_actions)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))                
        action = x * self.range_action / 2 + self.mean_action
        return action
        
        
# CRITIC Q
class Action_value_continuous(nn.Module):
    def __init__(self, env : gym.Env):
        n_obs = env.observation_space.shape[0]
        if len(env.observation_space.shape) > 1:
            raise NotImplementedError("Only works with 1D observation spaces.")
        dim_actions = env.action_space.shape
        super(Action_value_continuous, self).__init__()
        self.fc_obs1 = nn.Linear(n_obs, 32)
        self.fc_obs2 = nn.Linear(32, 32)
        
        self.fc_action1 = nn.Linear(*dim_actions, 32)
        self.fc_action2 = nn.Linear(32, 32)
        
        self.fc_global1 = nn.Linear(64, 32)
        self.fc_global2 = nn.Linear(32, 1)
    def forward(self, s, a):
        s = torch.relu(self.fc_obs1(s))
        # s = F.relu(self.fc_obs2(s))
        a = torch.relu(self.fc_action1(a))
        # a = F.relu(self.fc_action2(a))
        sa = torch.concat([s,a], dim = -1)
        sa = torch.relu(self.fc_global1(sa))
        sa = self.fc_global2(sa)
        return sa
            
                 
class DDPG(Agent):
    '''DDPG
    '''

    def __init__(self, env : gym.Env, agent_cfg : dict, train_cfg : dict):
        # Init : define RL agent variables/parameters from agent_cfg and metrics from train_cfg
        super().__init__(env = env, agent_cfg = agent_cfg, train_cfg = train_cfg)
        # Eventually add metrics relative to this particular agent
        self.metrics.append(ClassicalLearningMetrics(self))

        # Memory
        self.memory = Memory_episodic(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'prob'])
        
        # Build networks
        self.n_obs = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]   
             
        self.action_value = Action_value_continuous(env)
        self.action_value_target = deepcopy(self.action_value)
        self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params=self.action_value.parameters())
        
        self.policy = ContinuousPolicy(env)
        self.policy_target = deepcopy(self.policy)
        self.opt_policy = optim.Adam(lr = self.learning_rate_actor, params=self.policy.parameters())
                        
    @classmethod
    def get_space_types(cls):
        return ["continuous"]
       
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped numpy observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an (dim_action,) shaped action
        '''
        
        #Batching observation
        observation = torch.Tensor(observation)
        observations = observation.unsqueeze(0) # (1, observation_space)
        #Choose action
        action = self.policy(observations).detach()[0]
        n_actions = action.shape[0]
        noise = torch.normal(mean = 0, std=torch.ones((n_actions,)) * self.sigma)
        action = (action + noise).numpy()
        
        #Save metrics
        self.compute_metrics(mode = 'act')

        # Action
        return action


    def learn(self):
        '''Do one step of learning.
        '''
        values = dict()
        self.step += 1
        
        #Learn only at the end of episodes, and only every train_freq_episode episodes.
        if not self.memory.done:
            return
        self.episode += 1
        if self.episode % self.train_freq_episode != 0:
            return
        
        #Sample trajectories
        episodes = self.memory.sample(
            sample_size=self.sample_size,
            method = "random",
            )
        observations, actions, rewards, dones, next_observations = self.concat_episodes(episodes)
        observations, actions, rewards, dones, next_observations = self.shuffle_transitions(elements = [observations, actions, rewards, dones, next_observations])
        
        #Compute targets of Q using SARSA
        next_actions = self.policy_target(next_observations)
        Q_s_targets = self.compute_SARSA(rewards, next_observations, next_actions, dones, 
                                         q_output_is_scalar = True, 
                                         model="action_value_target",
                                            ).detach()
        
        #Update critic by minimizing the loss
        Q_s = self.action_value(observations, actions)
        criterion = nn.MSELoss()
        critic_loss = criterion(Q_s, Q_s_targets).mean()
        
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        
        #Update actor by maximizing Q values
        actions = self.policy(observations)
        Q_s_to_maximize = self.action_value(observations, actions)
        actor_loss = - Q_s_to_maximize.mean()  
        
        self.opt_policy.zero_grad()
        actor_loss.backward()
        self.opt_policy.step()
        
        #Update target networks
        for phi, phi_target in zip(self.action_value.parameters(), self.action_value_target.parameters()):
            phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data 
        for theta, theta_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            theta_target.data = self.tau * theta_target.data + (1-self.tau) * theta.data 
        
        #Save metrics
        values["critic_loss"] = critic_loss.detach().numpy()
        values["actor_loss"] = actor_loss.detach().numpy()
        values["value"] = Q_s_to_maximize.detach().mean().numpy()
        self.compute_metrics(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        self.memory.remember((observation, action, reward, done, next_observation, info))
            
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.compute_metrics(mode = 'remember', **values)