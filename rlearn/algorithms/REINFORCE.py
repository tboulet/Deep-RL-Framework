from copy import copy, deepcopy
import numpy as np
import math
import gym
import sys
import random
import matplotlib.pyplot as plt
from div.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.categorical import Categorical

from RL.MEMORY import Memory, Memory_episodic
from RL.CONFIGS import REINFORCE_CONFIG, REINFORCE_OFFPOLICY_CONFIG
from RL.METRICS import Metric_Performances, Metric_Time_Count, Metric_Total_Reward, MetricS_On_Learn
from rl_algos.AGENT import Agent


class REINFORCE(Agent):
    '''REINFORCE agent is an actor RL agent that performs gradient ascends on the estimated objective function to maximize.
    NN trained : Actor
    Policy used : On-policy
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : continuous (discrete not implemented)
    '''

    def __init__(self, actor : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Total_Reward, Metric_Time_Count]
        super().__init__(agent_cfg = REINFORCE_CONFIG, metrics = metrics)
        self.memory = Memory_episodic(MEMORY_KEYS = ['observation', 'action','reward', 'done'])
        
        self.policy = actor
        self.opt = optim.Adam(lr = 1e-4, params=self.policy.parameters())
        
                
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
        self.add_metric(mode = 'act')
        
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
        episodes = self.memory.sample(method='last', sample_size=self.n_episode)
        
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
                loss_mean += loss / self.n_episode
                
            #Backpropagate to improve policy
            self.opt.zero_grad()
            loss_mean.backward()
            self.opt.step()
            
        self.memory.__empty__()
    
        #Save metrics
        values["actor_loss"] = loss.detach().numpy()
        self.add_metric(mode = 'learn', **values)



    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        self.memory.remember((observation, action, reward, done))
                    
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.add_metric(mode = 'remember', **values)






class REINFORCE_OFFPOLICY(Agent):
    '''REINFORCE agent is an actor RL agent that performs gradient ascends on the estimated objective function to maximize.
    The offpolicy version add importances weights to keep an unbiased gradient.
    NN trained : Actor
    Policy used : Off-policy
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : continuous
    '''

    def __init__(self, actor : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Total_Reward, Metric_Time_Count]
        super().__init__(agent_cfg = REINFORCE_OFFPOLICY_CONFIG, metrics = metrics)
        self.memory = Memory_episodic(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'prob'])
        
        self.policy = actor
        self.opt = optim.Adam(lr = 1e-4, params=self.policy.parameters())
        
        
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped nummpy observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''
        with torch.no_grad():
            #Batching observation
            observation = torch.Tensor(observation)
            observations = observation.unsqueeze(0) # (1, observation_space)
            probs = self.policy(observations)        # (1, n_actions)
            distribs = Categorical(probs = probs)    
            actions = distribs.sample()
            action = actions.numpy()[0]
            
            #Save metrics
            self.add_metric(mode = 'act')
            
            # Action
            self.last_prob = probs[0, action]
            return action


    def learn(self):
        '''Do one step of learning.
        '''
        values = dict()
        self.step += 1
        
        #Learn only at the end of episodes, and only every train_freq_episodes episodes.
        if not self.memory.done:
            return
        self.episode += 1
        if self.episode % self.train_freq_episode != 0:
            return
        
        #Sample trajectories
        episodes = self.memory.sample(
            method = "last",
            sample_size=self.n_episode,
            )
            
        #Compute mean value of gradients over a batch
        for _ in range(self.gradient_steps):
            loss_mean = torch.tensor(0.)
            batch_size = len(episodes)
            
            for observations, actions, rewards, dones, old_probs in episodes:
                
                #Some actions dtype problem
                actions = actions.to(dtype = torch.int64)
                
                #Scaling the rewards
                if self.reward_scaler is not None:
                    rewards /= self.reward_scaler
                
                #Compute Gt the discounted sum of future rewards
                Gt_s = self.compute_MC(rewards)

                #Compute loss
                if self.J_method == "ratio_ln":
                    probs = self.policy(observations)   #(T, n_actions)
                    probs = torch.gather(probs, dim = 1, index = actions)[:,0]    #(T,)
                    log_probs = torch.log(probs)    #(T,)
                    
                    old_probs = old_probs[:, 0]
                    ratios = (probs / old_probs).detach()
                    # ratios = torch.clamp(ratios, 1 - self.epsilon_clipper, 1 + self.epsilon_clipper)
                    log_probs = torch.multiply(log_probs, ratios)
                    
                    loss = torch.multiply(log_probs, Gt_s)
                    loss = - torch.sum(loss)
                    loss_mean += loss / batch_size
                
                elif self.J_method == "ratio":
                    #Compute log probs
                    probs = self.policy(observations)   #(T, n_actions)
                    probs = torch.gather(probs, dim = 1, index = actions)[:,0]    #(T,)
                    ratios = probs / old_probs
                    ratios = torch.clamp(ratios, 1 - self.epsilon_clipper, 1 + self.epsilon_clipper)
                    
                    loss = - torch.multiply(ratios, Gt_s)
                    loss = torch.sum(loss)
                    loss_mean += loss / batch_size
                
                
                
            #Backpropagate to improve policy
            self.opt.zero_grad()
            loss_mean.backward()
            self.opt.step()
            
    
        #Save metrics
        values["actor_loss"] = loss.detach().numpy()
        self.add_metric(mode = 'learn', **values)



    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        return : metrics, a list of metrics computed during this remembering step.
        '''
        prob = self.last_prob
        self.memory.remember((observation, action, reward, done, prob, info))
            
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done, "prob" : prob}
        self.add_metric(mode = 'remember', **values)

    