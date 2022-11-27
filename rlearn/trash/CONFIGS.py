Q_TABLE_CONFIG = {"name" : "Q_TABLE",
    "gamma" : 0.99,
    "sample_size" : 64,
    "learning_rate" : 0.01,
    
    "exploration_timesteps" : 10000,
    "exploration_initial" : 1,
    "exploration_final" : 0.05,
    }

DQN_CONFIG = {"name" : "DQN",
    "gamma" : 0.99,
    "sample_size" : 256,
    "learning_rate" : 1e-3,
    "n_step" : 12,
    "algorithm_method" : "SARSA Expected n-step",   # "SARSA", "SARSAE" or "Q-Learning", "SARSAN", "SARSANE" or "Q-Learning n-step", "SARSA unbiased", "SARSAN unbiased"

    "history_lenght" : 1,   #To implement
    
    "reward_scaler" : 10,
    "tau" : 0.999,
    "clipping" : None,
    "train_freq" : 1,
    "gradients_steps" : 10,
     
    "learning_starts" : 2000,
    "exploration_timesteps" : 10000,
    "exploration_initial" : 1,
    "exploration_final" : 0.05,
    }

REINFORCE_CONFIG = {"name" : "REINFORCE",
    "learning_rate" : 1e-4,
    "gradient_steps" : 10,    #<!> for >1, this became off policy
    "gamma" : 0.99,
    "reward_scaler" : None,
    
    "train_freq_episode" : 1, 
    "n_episode" : 1,          
    }

REINFORCE_OFFPOLICY_CONFIG = {"name" : "REINFORCE_OFFPOLICY",
    "learning_rate" : 1e-4,
    "gradient_steps" : 4,   #<!> for >1, this became off policy
    "gamma" : 0.99,
    "reward_scaler" : None,
    
    "train_freq_episode" : 4, 
    "n_episode" : 8,
    
    "J_method" : "ratio_ln",   #ratio or ratio_ln     
    "epsilon_clipper" : 0.2,
    }

ACTOR_CRITIC_CONFIG = {"name" : "ACTOR_CRITIC",
    "learning_rate_actor" : 1e-2,
    "learning_rate_critic" : 1e-2,
    "compute_gain_method" : "GAE",
    "gamma" : 0.98,     
    "lmbda" : 0.98,
    "reward_scaler" : 100,
    "batch_size" : 2,        #TO IMPLEMENT. #Algorithm updates critic at every steps, and policy every batchsize steps, using the entire batch
    "gradient_steps_critic" : 4,
    "gradient_steps_policy" : 1,
    "tau" : 0.99,
    "clipping" : None,
    }

DDPG_CONFIG = {"name" : "DDPG",
    "learning_rate_actor" : 1e-3,
    "learning_rate_critic" : 1e-3,
    "gamma" : 0.99,     
    "reward_scaler" : 100,
    "sample_size" : 256,    
    "gradient_steps" : 1,
    "clipping" : None,
    "sigma" : 0.1,
    "train_freq" : 1,
    "tau" : 0.99,
    
    }

PPO_CONFIG = {"name" : "PPO",
    "learning_rate_actor" : 1e-4,
    "learning_rate_critic" : 1e-3,
    "gamma" : 0.99,
    "gae_lambda" : 0.98,
    
    "train_freq_episode" : 1,
    "n_episodes" : 32,
    "batch_size" : 128,
    "epochs" : 15,
    "n_step" : 10,

    "update_method" : "soft",
    "tau" : 0.99,
    "target_update_interval" : 10000,
    "reward_scaler" : 100,
    
    "epsilon_clipper" : 0.2,
    "c_critic" : 1,
    "c_entropy" : 0.01,
    }

DUMMY_CONFIG = dict()