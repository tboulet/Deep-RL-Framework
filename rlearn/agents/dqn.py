from copy import deepcopy
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim

from rlearn.core.memory import Memory_episodic
from rlearn.core.metrics import ClassicalLearningMetrics
from rlearn.agents import Agent

class DQN(Agent):

    implemented_algorithm_methods = ["SARSA", "SARSAE", "Q-Learning", "SARSAN", "SARSANE", "Q-Learning n-step", "SARSA unbiased", "SARSAN unbiased", ]
    
    @classmethod
    def get_supported_action_space_types(cls):
        return ["discrete"]
    

    def __init__(self, env : gym.Env, config : dict):

        super().__init__(env = env, config = config)

        # Memory
        self.memory = Memory_episodic(
            MEMORY_KEYS = ['observation', 'action','reward', 'done', 'prob'],
            max_memory_len=self.buffer_size,
            )

        # Build networks
        self.n_actions = env.action_space.n
        self.n_obs = self.env.observation_space.shape[0]
        if len(self.env.observation_space.shape) > 1:
            raise NotImplementedError("Only works with 1D observation spaces.")
        action_value = nn.Sequential(
                nn.Linear(self.n_obs, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_actions),
            )
        self.action_value = action_value
        self.action_value_target = deepcopy(action_value)
        self.opt = optim.Adam(lr = self.learning_rate, params=action_value.parameters())

    
    def act(self, observation, mask = None, training = True):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''

        #Batching observation
        observations = torch.Tensor(observation)
        observations = observations.unsqueeze(0) # (1, observation_space)
    
        # Q(s)
        Q = self.action_value(observations) # (1, action_space)
        action_greedy = torch.argmax(Q, axis = -1).detach().numpy()[0] 
        
        #Epsilon-greedy policy
        epsilon = self.get_eps()
        if np.random.rand() > epsilon:
            prob = 1-epsilon + epsilon/self.n_actions
            action = action_greedy
    
        else :
            action = torch.randint(size = (1,), low = 0, high = Q.shape[-1]).numpy()[0]     #Choose random action
            if action == action_greedy:
                prob = 1-epsilon + epsilon/self.n_actions
            else:
                prob = epsilon/self.n_actions
        
        #Save metrics            
        self.compute_metrics(mode = 'act')
    
        # Action
        self.last_prob = prob
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
        if self.episode % 1 != 0: # TODO : train_freq_episodes
            return

        #Sample trajectories
        episodes = self.memory.sample(
            sample_size=self.sample_size,
            method = "random",
            )

        Q_targets = list()
        for observations, actions, rewards, dones, probs in episodes:
            #Scaling the rewards
            if self.reward_scaler is not None:
                rewards = rewards / self.reward_scaler
            #Type errors
            actions = actions.to(dtype = torch.int64)
            #Compute Q targets
            next_observations = torch.roll(observations, shifts = -1, dims = 0)
            
            if self.algorithm_method == "SARSA":
                #SARSA : E_mu[Rt + g * (1-Dt) * Q(St+1, At+1)]
                next_actions = torch.roll(actions, shifts = -1, dims = 0)
                q_targets = self.compute_SARSA(rewards, next_observations, next_actions, dones, 
                                            model = 'action_value',
                                            importance_weights = None,
                                            )
            
            elif self.algorithm_method == "SARSA unbiased":
                #SARSA unbiased : E_mu[Rt + g * (1-Dt) * r_t+1 * Q(St+1, At+1)]
                next_actions = torch.roll(actions, shifts = -1, dims = 0)
                importance_weights = self.get_importance_weights(observations, actions, probs)
                q_targets = self.compute_SARSA(rewards, next_observations, next_actions, dones, 
                                               model = 'action_value',
                                               importance_weights = importance_weights,
                                               )
            
            elif self.algorithm_method in ["SARSAE", "Q-Learning"]:
                # #SARSA Expected : E_mu[Rt + g * (1-Dt) * Q(St+1, pi(St+1))]
                q_targets = self.compute_SARSA_Expected(rewards, next_observations, dones, 
                                                model = 'action_value',
                                                )
            
            elif self.algorithm_method == "SARSAN":
                #SARSAN : E_mu[Rt + g * Rt+1 + g² * Rt+2 + g^3 * Q(St+3, At+3)]
                q_targets = self.compute_SARSA_n_step(rewards, observations, actions, 
                                                model = 'action_value', 
                                                importance_weights=None
                                                )
                                            
            elif self.algorithm_method == "SARSAN unbiased":
                # #SARSAN unbiased : E_mu[Rt + g * r_t+1 * Rt+1 + g² * r_t+1*r_t+2 * Rt+2 + g^3 * r_t+1*r_t+2*t_t+3 * Q(St+3, At+3)]
                importance_weights = self.get_importance_weights(observations, actions, probs)
                q_targets = self.compute_SARSA_n_step(rewards, observations, actions, model = 'action_value', importance_weights=importance_weights)

                # SARSANE
            elif self.algorithm_method in ["SARSANE", "Q-Learning n-step"]:
                q_targets = self.compute_SARSA_n_step_Expected(observations, rewards, model = 'action_value', importance_weights=None)
            
            elif self.algorithm_method == "SARSANE unbiased":
                # SARSANE unbiased
                # TODO : implement
                raise NotImplementedError
            
            else:
                raise Exception(f"Unknown algorithm_method : {self.algorithm_method} not in {self.implemented_algorithm_methods}")
            
            Q_targets.append(q_targets)
            
            
        observations, actions, rewards, dones, probs = [torch.concat([episode[elem] for episode in episodes], axis = 0) for elem in range(len(episodes[0]))]
        Q_targets = torch.concat(Q_targets, axis = 0).detach()
        
        #Type errors...
        actions = actions.to(dtype = torch.int64)
        
        #Scaling the rewards
        if self.reward_scaler is not None:
            rewards = rewards / self.reward_scaler
            
        #Gradient descent on Q network
        criterion = torch.nn.MSELoss()
        for _ in range(self.gradients_steps):
            self.opt.zero_grad()
            Q_s = self.QSA(self.action_value, observations, actions)
            loss = criterion(Q_s, Q_targets)
            loss.backward(retain_graph = True)
            self.opt.step()
        
        #Update target network
        for phi, phi_target in zip(self.action_value.parameters(), self.action_value_target.parameters()):
            phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data    
       
        #Save metrics
        values["critic_loss"] = loss.detach().numpy()
        values["value"] = Q_s.mean().detach().numpy()
        self.compute_metrics(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        prob = self.last_prob
        self.memory.remember((observation, action, reward, done, prob))        
        
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.compute_metrics(mode = 'remember', **values)
    

    def get_eps(self):
        """Get the current epsilon value"""
        return max(self.exploration_final, self.exploration_initial + (self.exploration_final - self.exploration_initial) * (self.step / self.exploration_timesteps))
    
    
    