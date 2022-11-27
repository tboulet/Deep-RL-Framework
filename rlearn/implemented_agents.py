from rlearn.algorithms import DQN, RandomAgent
import gym

agents_map = {
    "random": RandomAgent,
    "DQN": DQN,
}


def get_env_type(env : gym.Env):
    if isinstance(env.action_space, gym.spaces.Discrete) and isinstance(env.observation_space, gym.spaces.Discrete):
        return 'discrete'
    elif isinstance(env.action_space, gym.spaces.Box) and isinstance(env.observation_space, gym.spaces.Box):
        return 'continuous'
    elif isinstance(env.action_space, gym.spaces.Discrete) and isinstance(env.observation_space, gym.spaces.Box):
        return 'obs-continuous'
    elif isinstance(env.action_space, gym.spaces.Box) and isinstance(env.observation_space, gym.spaces.Discrete):
        return 'action-continuous'
    else:
        raise Exception(f"Env {env} has not recognized action and observation spaces : Action: {env.action_space} and Obs: {env.observation_space}")

    
def make_agent(agent_name : str, agent_cfg : dict, train_cfg : dict, env : gym.Env):

    if agent_name not in agents_map:
        raise Exception(f"Agent {agent_name} not found in implemented agents in implemented_agents.py")
    AgentCls = agents_map[agent_name]
    
    env_type = get_env_type(env)
    if not hasattr(AgentCls, 'space_types'):
        raise Exception(f"Agent {agent_name} has no space_types attribute defined")
    if env_type not in AgentCls.space_types:
        raise Exception(f"Agent {agent_name} does not support env type {env_type}  (obs space {env.observation_space} and action space {env.action_space} )")
    
    agent = AgentCls(env = env, agent_cfg = agent_cfg, train_cfg = train_cfg)
    return agent