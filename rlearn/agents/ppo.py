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

class PPO(Agent):
    '''PPO updates its networks without changing too much the policy, which increases stability.
    NN trained : Actor Critic
    Policy used : Off-policy
    Stochastic : Yes
    Actions : discrete (continuous not implemented)
    States : continuous (discrete not implemented)
    '''

    def __init__(self, env : gym.Env, agent_cfg : dict, train_cfg : dict):
        # Init : define RL agent variables/parameters from agent_cfg and metrics from train_cfg
        super().__init__(env = env, agent_cfg = agent_cfg, train_cfg = train_cfg)
        # Additional metrics for PPO
        self.metrics.append(ClassicalLearningMetrics(self))
        
        # Memory
        self.memory = Memory_episodic(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'prob'])
        
        # Build networks
        self.n_actions = env.action_space.n
        self.n_obs = env.observation_space.shape[0]
        if len(env.observation_space.shape) > 1:
            raise NotImplementedError("Only works with 1D observation spaces.")
        
        self.state_value = nn.Sequential(
                nn.Linear(self.n_obs, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        self.state_value_target = deepcopy(self.state_value)
        self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params=self.state_value.parameters())
        
        self.policy = nn.Sequential(
                nn.Linear(self.n_obs, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_actions),
                nn.Softmax(dim=-1)
            )
                        
    @classmethod
    def get_space_types(cls):
        return ["semi-continuous"]
    
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped numpy observation.
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
        self.last_prob = probs[0, action].detach()
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
            sample_size=self.num_episodes,
            )
        
        #Compute A_s and V_s estimates and concatenate trajectories. 
        advantages = list()
        V_targets = list()
        for observations, actions, rewards, dones, probs in episodes:
            #Scaling the rewards
            if hasattr(self, "reward_scaler") and self.reward_scaler is not None:
                rewards = rewards / self.reward_scaler
            #Compute V and A on one episode
            A_episode = self.compute_GAE(rewards, observations)
            V_targets_episode = self.compute_TD_n_step(rewards, observations, model = 'state_value_target')
            advantages.append(A_episode)
            V_targets.append(V_targets_episode)
        advantages = torch.concat(advantages, axis = 0).detach()
        V_targets = torch.concat(V_targets, axis = 0).detach()
        observations, actions, rewards, dones, probs = self.concat_episodes(episodes)
        
        #Shuffling data
        observations, actions, rewards, dones, probs, advantages, V_targets = \
            self.shuffle_transitions(elements = [observations, actions, rewards, dones, probs, advantages, V_targets])
        
        #Type bug fixes
        actions = actions.to(dtype = torch.int64)
        rewards = rewards.float()
        
        #We perform gradient descent on n_epochs epochs on T datas with minibatch of size batch_size <= T. Where T is the sum of the lenghts of all episodes.
        policy_new = deepcopy(self.policy)
        opt_policy = optim.Adam(lr = self.learning_rate_actor, params=policy_new.parameters())           
        n_batch = math.ceil(len(observations) / self.batch_size)
    
        for _ in range(self.gradient_steps):
            for i in range(n_batch):
                #Batching data
                observations_batch = observations[i * self.batch_size : (i+1) * self.batch_size]
                actions_batch = actions[i * self.batch_size : (i+1) * self.batch_size]
                probs_batch = probs[i * self.batch_size : (i+1) * self.batch_size]
                advantages_batch = advantages[i * self.batch_size : (i+1) * self.batch_size]
                V_targets_batch = V_targets[i * self.batch_size : (i+1) * self.batch_size]

                #Objective function : J_clip = min(r*A, clip(r,1-e,1+e)A)  where r = pi_theta_new/pi_theta_old and A advantage function
                pi_theta_new_s_a = policy_new(observations_batch)
                pi_theta_new_s   = torch.gather(pi_theta_new_s_a, dim = 1, index = actions_batch)
                ratio_s = pi_theta_new_s / probs_batch
                ratio_s_clipped = torch.clamp(ratio_s, 1 - self.ratio_clipper, 1 + self.ratio_clipper)
                J_clip = torch.minimum(ratio_s * advantages_batch, ratio_s_clipped * advantages_batch).mean()

                #Error on critic : L = L(V(s), V_target)   with V_target = r + gamma * (1-d) * V_target(s_next)
                V_s = self.state_value(observations_batch)
                critic_loss = nn.MSELoss()(V_s, V_targets_batch).mean()
                
                #Entropy : H = sum_a(- log(p) * p)      where p = pi_theta(a|s)
                log_pi_theta_s_a = torch.log(pi_theta_new_s_a)
                pmlogp_s_a = - log_pi_theta_s_a * pi_theta_new_s_a
                H_s = torch.sum(pmlogp_s_a, dim = 1)
                H = H_s.mean()
                            
                #Total objective function
                J = J_clip - self.c_value * critic_loss + self.c_entropy * H
                loss = - J
                
                #Gradient descend
                opt_policy.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward(retain_graph = True)
                opt_policy.step()
                self.opt_critic.step()
                
        
        #Update policy
        self.policy = deepcopy(policy_new)
        
        #Update target network
        if self.update_method == "periodic":
            if self.step % self.target_update_interval == 0:
                self.state_value_target = deepcopy(self.state_value)
        elif self.update_method == "soft":
            for phi, phi_target in zip(self.state_value.parameters(), self.state_value_target.parameters()):
                phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data    
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Save metrics
        values["critic_loss"] = critic_loss.detach().numpy()
        values["J_clip"] = J_clip.detach().numpy()
        values["value"] = V_s.mean().detach().numpy()
        values["entropy"] = H.mean().detach().numpy()
        self.compute_metrics(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        prob = self.last_prob.detach()
        self.memory.remember((observation, action, reward, done, prob))
            
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.compute_metrics(mode = 'remember', **values)