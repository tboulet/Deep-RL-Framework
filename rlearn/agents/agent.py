from abc import ABC, abstractmethod, abstractstaticmethod

import torch
import wandb
from random import randint
from rlearn.metrics import get_metrics
import gym
from torch import Tensor
from typing import List

class Agent(ABC):
    
    def __init__(self, env : gym.Env, agent_cfg : dict = dict(), train_cfg : dict = dict()):
        self.step = 0
        self.episode = 0
        self.env = env
        self.config = agent_cfg if agent_cfg is not None else dict()
        self.train_cfg = train_cfg if train_cfg is not None else dict()
        self.metrics = [Metric(self) for Metric in get_metrics(train_cfg["metrics"])]
        for name, value in self.config.items():
            setattr(self, name, value)
        self.metrics_saved = list()
        
    @abstractmethod
    def act(self, obs : Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, **kwargs):
        pass
    
    @abstractstaticmethod
    def get_space_types() -> List[str]:
        raise Exception("You must implement this method in your agent class")
    
    def add_metric(self, mode, **values):
        if mode == 'act':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_act(**values))
        if mode == 'remember':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_remember(**values))
        if mode == 'learn':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_learn(**values))    
    
    def log_metrics(self):
        for metric in self.metrics_saved:
            if self.train_cfg["wandb"]:
                wandb.log(metric, step = self.step)
            if self.train_cfg["tb"]:
                for key, value in metric.items():
                    self.tb_writer.add_scalar(key, value, self.step)
            if self.train_cfg["print"]:
                if len(metric) > 0:
                    print(metric)
            if self.train_cfg["dump"]:
                pass
        self.metrics_saved = list()
    
    
    
    def concat_episodes(self, episodes : List[Tensor]):
        """Concatenate a list of episodes (of shape (n_episodes, n_elem, T, *elem_dims)) into a single tuple of shape (n_elem, n_episodes * T, *elem_dims)

        Args:
            episodes (List[Tensor]): a list of episodes, usually obtained by the .sample() method of the episodic replay buffer

        Returns:
            tuple: a tuple of tensors, each of shape (n_episodes * T, *elem_dims)
        """
        return [torch.concat([episode[elem] for episode in episodes], axis = 0) for elem in range(len(episodes[0]))]
    
    
    
    def shuffle_transitions(self, elements : List[Tensor]):
        indexes = torch.randperm(len(elements[0]))
        return [element[indexes] for element in elements]
    
    
    
    def QSA(self, model, observations, actions, q_output_is_scalar = False):
        """Compute the scalar Q values of a given model.

        Args:
            model (nn.Module or str): The model used to compute Q values. If str, it must be the name of an attribute of the agent.
            observations (tensor): a (T, *dims) shaped tensor representing observations
            actions (tensor): a (T, 1) (discrete) or (T, *dim_actions) (continuous) shaped tensor representing actions
            q_output_is_scalar (bool, optional): Weather the Q network works in a "Q(s) = [Q(s,a) for a in A]" way of in a "Q(s,a) = scalar" way. Defaults to False.

        Returns:
            tensor: a (T, 1) shaped tensor representing Q values
        """
        if isinstance(model, str):
            model = getattr(self, model)
        if q_output_is_scalar:
            return model(observations, actions)
        else:
            Q_s_a = model(observations)
            Q_s = Q_s_a.gather(dim = 1, index = actions)
            return Q_s
    
    def pi(self, observations):
        '''Return actions corresponding to current evaluated policy.
        observations : a (T, *dims) shaped tensor representing observations
        return : a (T, *dim_actions) shaped tensor reprensenting actions taken
        '''
        return torch.Tensor([self.act(observation = observation, training = False) for observation in observations.numpy()]).unsqueeze(dim = -1).long()
    
    def compute_TD(self, rewards, next_observations, dones, model = 'state_value', importance_weights = None):
        '''Compute the 1 step TD estimates V(s) of state values : V_pi(St) = E_mu[imp_weights_t * (Rt + g * V(St+1))]
        rewards, next_observations, dones : (T, 1) shaped torch tensors representing rewards, next_observations, dones
        model : the name of the attribute of agent used for computing state values, in ('state_value', 'state_value_target')
        importance_weights : the ratio of sampling policy probs over evaluated policy probs, which allow unbiased offline learning. If None, no importance weights is applied.
        return : a (T, 1) shaped torch tensor representing state values
        '''
        model = getattr(self, model)
        values = rewards + (1 - dones) * self.gamma * model(next_observations)
        if importance_weights is None:
            return values
        else:
            return values * importance_weights
    
    def compute_SARSA(self, rewards, next_observations, next_actions, dones, model = 'action_value', importance_weights = None, q_output_is_scalar = False):
        '''Compute the 1 step SARSA estimates Q(s,a) of action values over one episode: Q_pi(St, At) = E_mu[Rt + g * (1-Dt) * r_t+1 * Q_pi(St+1, At+1)]
        observations, actions, rewards, next_observations, next_actions, dones : (T, *dims) shaped torch tensors sampled with policy mu
        model : the name of the attribute of agent used for computing state values, in ('action_value', 'action_value_target')
        importance_weights : the ratio of sampling policy probs over evaluated policy probs, which allow unbiased offline learning. If None, no importance weights is applied.
        return : a (T, 1) shaped torch tensor representing action values
        '''
        model = getattr(self, model)
        Q_s_future = self.QSA(model, next_observations, next_actions, q_output_is_scalar=q_output_is_scalar)
        
        if importance_weights is None:
            return rewards + (1 - dones) * self.gamma * Q_s_future
        else:
            next_importance_weights = torch.roll(importance_weights, shifts = -1, dims = 0)
            return rewards + (1 - dones) * self.gamma * next_importance_weights * Q_s_future
    
    def compute_SARSA_Expected(self, rewards, next_observations, dones, model = 'action_value'):
        '''Compute the 1 step SARSA-Expected estimates Q(s,a) of action values : Q_pi(St, At) = E_mu[Rt + g * (1-Dt) * Q_pi(St+1, pi(St+1))]
        NOTE : This is an easy-to-implement but more variant version of SARSAE. In the ideal case you sum the Q values and weight by the probs.
        observations, actions, rewards, next_observations, dones : (T, *dims) shaped torch tensors sampled with policy mu
        model : the name of the attribute of agent used for computing state values, in ('action_value', 'action_value_target')
        return : a (T, 1) shaped torch tensor representing action values
        '''
        model = getattr(self, model)
        next_actions = self.pi(next_observations)
        Q_s_future = self.QSA(model, next_observations, next_actions)
        
        return rewards + (1 - dones) * self.gamma * Q_s_future
    
    def compute_SARSA_n_step(self, rewards, observations, actions, model = 'action_value', importance_weights = None):
        '''Compute the n step SARSA estimates Q(s,a) of action values over one episode: Q_pi(St, At) = E_mu[Rt + g * r_t+1 * Rt+1 + g² * r_t+1*r_t+2 * Rt+2 + g^3 * r_t+1*r_t+2*r_t+3 * Q(St+3, At+3)]
        observations, actions, rewards, next_observations, next_actions, dones : (T, *dims) shaped torch tensors sampled with policy mu
        importance_weights : the ratio of sampling policy probs over evaluated policy probs, which allow unbiased offline learning. If None, no importance weights is applied.
        model : the name of the attribute of agent used for computing state values, in ('action_value', 'action_value_target')
        return : a (T, 1) shaped torch tensor representing action values
        '''
        model = getattr(self, model)
        rewards = rewards[:, 0]
        q_values = self.QSA(model, observations, actions)[:, 0]
        T = len(rewards)
        n = self.n_step
        Q = [None for _ in range(T)]
        U = 0
        t = T - 1
        if importance_weights is None:
            while t >= 0:
                if t >= T - n:
                    U = rewards[t] + self.gamma * U
                    Q[t] = U
                else:
                    U = rewards[t] + self.gamma * U - self.gamma ** n * rewards[t+n]
                    Q[t] = U + self.gamma ** n * q_values[t+n]
                t -= 1
                
        else:
            importance_weights = importance_weights[:,0]
            next_imp_weight_t = 1
            while t >= 0:
                ratio_products = torch.prod(importance_weights[t+1:t+1+n], 0)
                if t >= T - n:
                    U = rewards[t] + self.gamma * U * next_imp_weight_t
                    Q[t] = U
                else:   
                    U = rewards[t] + self.gamma * U * next_imp_weight_t - self.gamma ** n * ratio_products * rewards[t+n]
                    Q[t] = U + self.gamma ** n * ratio_products * q_values[t+n]
                    
                next_imp_weight_t = importance_weights[t]   #r_t+1
                t -= 1
                
                
        Q = torch.Tensor(Q).unsqueeze(-1)
        return Q    
    
    def compute_SARSA_n_step_Expected(self, observations, rewards, model = 'action_value', importance_weights = None):
        '''Compute the n step SARSA-Expected estimates Q(s,a) of action values over one episode: Q_pi(St, At) = E_mu[Rt + g * r_t+1 * Rt+1 + g² * r_t+1*r_t+2 * Rt+2 + g^3 * r_t+1*r_t+2 * Q(St+3, pi(St+3))]
        observations, rewards : (T, *dims) shaped torch tensors sampled with policy mu
        importance_weights : the ratio of sampling policy probs over evaluated policy probs, which allow unbiased offline learning. If None, no importance weights is applied.
        model : the name of the attribute of agent used for computing state values, in ('action_value', 'action_value_target')
        return : a (T, 1) shaped torch tensor representing action values
        '''
        model = getattr(self, model)
        rewards = rewards[:, 0]
        actions_pi = self.pi(observations)
        q_values = self.QSA(model, observations, actions_pi)[:, 0]
        T = len(rewards)
        n = self.n_step
        Q = [None for _ in range(T)]
        U = 0
        t = T - 1
        if importance_weights is None:
            while t >= 0:
                if t >= T - n:
                    U = rewards[t] + self.gamma * U
                    Q[t] = U
                else:
                    U = rewards[t] + self.gamma * U - self.gamma ** n * rewards[t+n]
                    Q[t] = U + self.gamma ** n * q_values[t+n]
                t -= 1
                
        else:
            importance_weights = importance_weights[:,0]
            next_imp_weight_t = 1
            while t >= 0:
                ratio_products = torch.prod(importance_weights[t+1:t+n], 0)
                if t >= T - n:
                    U = rewards[t] + self.gamma * U * next_imp_weight_t
                    Q[t] = U
                else:   
                    U = rewards[t] + self.gamma * U * next_imp_weight_t - self.gamma ** n * ratio_products * importance_weights[t+n] * rewards[t+n]
                    Q[t] = U + self.gamma ** n * ratio_products * q_values[t+n]
                    
                next_imp_weight_t = importance_weights[t]   #r_t+1
                t -= 1
                
                
        Q = torch.Tensor(Q).unsqueeze(-1)
        return Q  
    
    
    def compute_MC(self, rewards):
        '''Compute the sums of future rewards (discounted) over one episode.
        It is the Monte Carlo estimation of a state value : Rt + g * Rt+1 + ... + g^T-t * RT
        rewards : a (T, 1) shaped torch tensor representing rewards
        return : a (T, 1) shaped torch tensor representing discounted sum of future rewards
        '''
        #We compute the discounted sum of the next rewards dynamically.
        T = len(rewards)
        rewards = rewards[:, 0]
        future_rewards =  [None for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:   
            future_rewards[t] = rewards[t] + self.gamma * future_rewards[t+1]
            t -= 1
        future_rewards.pop(-1)
        future_rewards = torch.Tensor(future_rewards).unsqueeze(-1)          
        return future_rewards
    
    
    def compute_TD_n_step(self, rewards, observations, model = "state_value"):
        '''Compute the n_step TD estimates V(s) of state values over one episode, where n_step is an int attribute of agent.
        It follows the Temporal Difference relation V(St) = Rt + g*Rt+1 + ... + g^n-1 * Rt+n-1 + g^n * V(St+n)
        rewards : a (T, 1) shaped torch tensor representing rewards
        observations : a (T, *dims) shaped torch tensor representing observations
        model : the name of the attribute of agent used for computing state values, in ('state_value', 'state_value_target')
        return : a (T, 1) shaped torch tensor representing state values
        '''
        n = self.n_step
        
        #We compute the discounted sum of the n next rewards dynamically.
        T = len(rewards)
        rewards = rewards[:, 0]
        n_next_rewards =  [0 for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:   
            if t >= T - n:
                n_next_rewards[t] = rewards[t] + self.gamma * n_next_rewards[t+1]
            else:
                n_next_rewards[t] = rewards[t] + self.gamma * n_next_rewards[t+1] - (self.gamma ** n) * rewards[t+n]
            t -= 1
        n_next_rewards.pop(-1)
        n_next_rewards = torch.Tensor(n_next_rewards).unsqueeze(-1)

        #We compute the state value, and shift them forward in order to add them or not to the estimate.
        model = getattr(self, model)
        state_values = model(observations)
        state_values_to_add = torch.concat((state_values, torch.zeros(n, 1)), axis = 0)[n:]
        
        V_targets = n_next_rewards + state_values_to_add
        return V_targets        
    
    def compute_GAE(self, rewards, observations):
        '''Compute the Generalized Advantage Estimator (GAE) of advantage function over one episode.
        rewards : a (T, 1) shaped torch tensor representing rewards
        observations : a (T, *dims) shaped torch tensor representing observations
        return : a (T, 1) shaped torch tensor representing advantages functions
        '''
        T = len(rewards)
        rewards = rewards[:, 0].numpy()
        values = self.state_value(observations)[:, 0].detach().numpy()
        #We compute the TD residuals delta_t = Rt + (1-Dt) * g * V(St+1) - V(St)
        deltas = list()
        for t in range(T-1):
            deltas.append(rewards[t] + self.gamma * values[t+1] - values[t])
        deltas.append(rewards[T-1] - values[T-1])   #Last residual is just RT - V(ST) since this is the end of episode.
        #We compute dynamically the GAE
        A_GAE = [None for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:
            A_GAE[t] = deltas[t] + self.gamma * self.lam * A_GAE[t + 1]
            t -= 1
        A_GAE.pop(-1)
        A_GAE = torch.Tensor(A_GAE).unsqueeze(-1)                
        return A_GAE
       
    def get_importance_weights(self, observations, actions, probs):
        """Compute importance weights, ie [pi(a|s) / mu(a|s) for each (s,a) in the batch]
        observations, actions : (T, *dims) shaped torch tensors sampled with policy mu
        probs : (T, 1) shaped torch tensor of probabilities of actions sampled with policy mu
        return : (T, 1) shaped torch tensor of importance weights
        """
        q_values = self.action_value(observations)
        best_actions_for_Q = torch.argmax(q_values, dim = 1, keepdim=True)
        greedyQ_probs = (actions == best_actions_for_Q).float()
        importance_weights = greedyQ_probs / probs
        return importance_weights   
       
       

#Use the following agent as a model for minimum restrictions on AGENT subclasses :
class ExampleAgent(Agent):
    '''A n example agent evolving in a discrete environment.
    n_actions : int, n of action space
    '''
    def __init__(self, n_actions):
        super().__init__(metrics=get_metrics(["MetricS_On_Learn_Numerical", "Metric_Performances"])) #Choose metrics here
        self.n_actions = n_actions  #For RandomAgent only
        raise
    
    def act(self, obs):
        #Choose action here
        ...
        action = randint(0, self.n_actions - 1)
        #Save metrics
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'act', **values)
        
        return action
    
    def learn(self):
        #Learn here
        ...
        #Save metrics
        self.step += 1
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'learn', **values)
    
    def remember(self, *args):
        #Save kwargs in memory here
        ... 
        #Save metrics
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'remember', **values)