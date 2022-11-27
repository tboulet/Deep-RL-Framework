# Generate function approximation adapted to the environment and to the function of the network.
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces as spaces
import numpy as np


    
    
# class Linear_function_approximation_V(nn):
#     '''General function approximation'''

#     def __init__(self, *dims):
#         super().__init__()
#         self.dims = dims
        
#         weights = torch.Tensor(*dims)
#         self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
#         nn.init.kaiming_uniform_(self.weights, a=2) # weight init

#     def forward(self, observations):
#         x_repr = self.state_representation(observations)
#         return torch.mm(x_repr, self.weights.t()).sum()
    
# class Tabular_V(Linear_function_approximation_V):
#     def __init__(self, state_n):
#         super().__init__(state_n)
#         self.state_n = state_n
#     def state_representation(observations):
#         n_obs = observations.shape[0]
        
# class State_aggregation_value(nn):
#     '''State aggregation function approximation. The state or state/action space is divided in boxes.
#     '''
#     def __init__(self, low_state, high_state, low_action = None, high_action = None, subparts = 2):
#         super().__init__()
#         self.subparts = subparts
#         pass
        


def create_functions(env):
    
    action_value = None
    state_value = None
    actor = None
    q_table = None
    v_table = None
    actor_table = None
    
    # S = DISCRETE, A = DISCRETE
    if isinstance(env.action_space, spaces.Discrete) and isinstance(env.observation_space, spaces.Discrete):
        print(f"\nCreation of networks for gym environment {env}. Type of spaces :\nS = DISCRETE\nA = DISCRETE\n")
    
        n_obs = env.observation_space.n
        n_actions = env.action_space.n
        
        #Q-TABLE Q
        class Q_table_discrete:
            def __init__(self, n_obs, n_actions):
                self.q_values = np.zeros((n_obs, n_actions))
            def __call__(self, obs, action = None):
                '''Call Q values.
                obs : an int representing a state
                action : an int reprensenting an action
                return : a () shaped array representing Q(s,a), or a (n_actions,) shaped array representing [Q(s,a) for a] if a is None
                '''
                if action is None:
                    return self.q_values[obs, :]
                else:
                    return self.q_values[obs, action]
        q_table = Q_table_discrete(n_obs, n_actions)
        
        #V-TABLE V
        class V_table_discrete:
            def __init__(self, n_obs):
                self.v_values = np.zeros((n_obs,))
            def __call__(self, obs):
                '''Call V values.
                obs : an int representing a state
                return : a () shaped array representing V(s)
                '''
                return self.q_values[obs]
        v_table = V_table_discrete(n_obs)
        
        #ACTOR TABLE
        pass
    
    
    
    
    
        
    # S = CONTINUOUS, A = DISCRETE
    elif isinstance(env.action_space, spaces.Discrete) and isinstance(env.observation_space, spaces.Box):
        print(f"\nCreation of networks for gym environment {env}. Type of spaces :\nS = CONTINUOUS\nA = DISCRETE\n")
    
        n_obs, *args = env.observation_space.shape
        n_actions = env.action_space.n

        #ACTOR PI
        actor = nn.Sequential(
                nn.Linear(n_obs, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
                nn.Softmax(dim=-1),
            )

        #CRITIC Q
        action_value = nn.Sequential(
                nn.Linear(n_obs, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions),
            )

        #STATE VALUE V
        state_value = nn.Sequential(
                nn.Linear(n_obs, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        
        
        
        
        
        
    # S = CONTINUOUS, A = CONTINUOUS
    elif isinstance(env.action_space, spaces.Box) and isinstance(env.observation_space, spaces.Box):
        print(f"\nCreation of networks for gym environment {env}. Type of spaces :\nS = CONTINUOUS\nA = CONTINUOUS\n")
    
        n_obs, *args = env.observation_space.shape
        dim_actions = len(env.action_space.shape)
        
        #ACTOR PI
        class Actor_continuous(nn.Module):
            def __init__(self):
                super(Actor_continuous, self).__init__()
                a_high = env.action_space.high
                a_low = env.action_space.low
                self.range_action = torch.Tensor(a_high - a_low)
                self.mean_action = torch.Tensor((a_high + a_low)/2)
                self.fc1 = nn.Linear(n_obs, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, dim_actions)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                # x = F.relu(self.fc2(x))
                x = torch.tanh(self.fc3(x))                
                action = x * self.range_action / 2 + self.mean_action
                return action
        actor = Actor_continuous()

        #ACTION VALUE Q
        class Action_value_continuous(nn.Module):
            def __init__(self):
                super(Action_value_continuous, self).__init__()
                self.fc_obs1 = nn.Linear(n_obs, 32)
                self.fc_obs2 = nn.Linear(32, 32)
                
                self.fc_action1 = nn.Linear(dim_actions, 32)
                self.fc_action2 = nn.Linear(32, 32)
                
                self.fc_global1 = nn.Linear(64, 32)
                self.fc_global2 = nn.Linear(32, 1)
            def forward(self, s, a):
                s = F.relu(self.fc_obs1(s))
                # s = F.relu(self.fc_obs2(s))
                a = F.relu(self.fc_action1(a))
                # a = F.relu(self.fc_action2(a))
                sa = torch.concat([s,a], dim = -1)
                sa = F.relu(self.fc_global1(sa))
                sa = self.fc_global2(sa)
                return sa
        action_value = Action_value_continuous()

        #STATE VALUE V
        class State_value_continuous(nn.Module):
            def __init__(self):
                super(State_value_continuous, self).__init__()
                self.fc1 = nn.Linear(n_obs, 32)
                self.fc2 = nn.Linear(32, 32)
                self.fc3 = nn.Linear(32, 1)
            def forward(self, s):
                s = F.relu(self.fc1(s))
                s = F.relu(self.fc2(s))
                s = self.fc3(s)
                return s
        state_value = State_value_continuous()

        
        
        
        
        
        
    else:
        raise Exception("Unknow type of gym env.")
        
    return {"actor" : actor,
            "state_value" : state_value,
            "action_value" : action_value,
            "q_table" : q_table,
            "v_table" : v_table,
            }