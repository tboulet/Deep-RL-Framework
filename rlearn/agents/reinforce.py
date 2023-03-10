from copy import copy, deepcopy
import numpy as np
import math
import gym
import sys
import random
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from copy import copy, deepcopy
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

from copy import deepcopy
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.categorical import Categorical

from rlearn.core.memory import Memory_episodic
from rlearn.core.metrics import ClassicalLearningMetrics
from rlearn.agents import Agent


class REINFORCE(Agent):
    '''REINFORCE agent is an actor RL agent that performs gradient ascends on the estimated objective function to maximize.
    NN trained : Actor
    Policy used : On-policy
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : continuous (discrete not implemented)
    '''

    @classmethod
    def get_supported_action_space_types(cls):
        return ["discrete"]
    

    def __init__(self, env : gym.Env, config : dict):

        super().__init__(env = env, config = config)

        # Create memory
        self.memory = Memory_episodic(
            MEMORY_KEYS = ['observation', 'action','reward', 'done'],
            )
                
        # Build networks
        self.n_actions = env.action_space.n
        self.n_obs = env.observation_space.shape[0]
        self.policy = nn.Sequential(
                nn.Linear(self.n_obs, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_actions),
                nn.Softmax(dim=-1)
            )
        self.opt = optim.Adam(lr = 1e-4, params=self.policy.parameters())
    
    @classmethod
    def get_space_types(self):
        return ["semi-continuous"]  # TODO : continuous, discrete
                
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped nummpy observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''
        
        #Batching observation
        observation = torch.Tensor(observation)
        observations = observation.unsqueeze(0) # (1, observation_space)
        probs = self.policy(observations)        # (1, n_actions)
        distribs = Categorical(probs = probs)    
        actions = distribs.sample()
        action = actions.numpy()[0]
        
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
        episodes = self.memory.sample(method='last', sample_size=self.num_episodes)
        
        #Compute mean value of gradients over a batch of episodes
        for _ in range(self.gradient_steps):
            loss_mean = torch.tensor(0.)
            
            for observations, actions, rewards, dones in episodes:
                
                #Some actions dtype problem
                actions = actions.to(dtype = torch.int64)
                #Scaling the rewards
                if self.reward_scaler is not None:
                    rewards /= self.reward_scaler
                
                #Compute Gt the discounted sum of future rewards
                Gt_s = self.compute_MC(rewards)
                                
                #Compute log probs
                probs = self.policy(observations)   #(T, n_actions)
                probs = torch.gather(probs, dim = 1, index = actions)   #(T, 1)
                log_probs = torch.log(probs)[:,0]     #(T,)
                
                #Compute loss = -sum_t( G_t * log_proba_t ) and add it to mean loss
                loss = torch.multiply(log_probs, Gt_s)
                loss = - torch.sum(loss)
                loss_mean += loss / self.num_episodes
                
            #Backpropagate to improve policy
            self.opt.zero_grad()
            loss_mean.backward()
            self.opt.step()
            
        self.memory.__empty__()
    
        #Save metrics
        values["actor_loss"] = loss.detach().numpy()
        self.compute_metrics(mode = 'learn', **values)



    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        self.memory.remember((observation, action, reward, done))
                    
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.compute_metrics(mode = 'remember', **values)