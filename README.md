# **<project>**

# A Deep Reinforcement Learning framework : RLearn

RLearn is a framework for Deep Reinforcement Learning made by myself. I implements diverse RL algorithms and try to make them learn on various environnements. It is based on the [OpenAI Gym](https://gym.openai.com/) and [pytorch](https://pytorch.org/).

<p align="center">
  <img src="assets/gym_envs.jpg" alt="Some RL environnements" width="60%"/>
</p>

# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Agents](#agents)

# Installation

**Advised : create a virtual env :**

```bash
python -m venv venvRLEARN
source venvRLEARN/bin/activate # on linux
venvRLEARN\Scripts\activate    # on windows
```

**Option 1 : Install the package from PyPI:**

```bash
pip install rlearn (WIP, not available yet)
```

**Option 2 : Install the package from GitHub:**

```bash
pip install git+https://github.com/tboulet/Deep-RL-Framework.git
```

**Option 3 : Install the package in local:**

```bash
git clone git@github.com:tboulet/Deep-RL-Framework.git
cd Deep-RL-Framework
pip install -e .
```

**Install the dependencies:**

```bash
# Clone and go inside the repository
pip install -r requirements.txt
```


# Usage

**For training algorithms inside this repository:**

```bash
python train.py --agent <agent> --env <env>
```

Please note that your Gym environnement must be registered in order to be used.

**Fields:**

Required fields (all other are optional):
- agent : the name of your algorithm (e.g. DQN, PPO, random, ...). See the list of available agents below.
- env : the name of your environnement (e.g. CartPole-v0, LunarLander-v2, ...). The env must be registered as a Gym environnement.

Training fields:
- train : whether to train the agent or not.
- load : whether to load the agent or not.  (WIP)
- load_path : the path to the agent to load. By default, the most advanced (in terms of timesteps spend training on the env) agent will be loaded.

Configurations: The default configs are in configs/default/
- agent-cfg : the path to the config file of the agent. A default config file for the agent will be used if not specified.
- train-cfg : the path to the config file of the training. A default config file for the training will be used if not specified.

Feedback and logging:
- wandb : whether to use wandb or not. If true you must have specified your IDs in the settings.py file (see template at templates/settings.py).
- tb : whether to use tensorboard or not.
- n_render : the frequency (one episode each n_render) of episodes to render.

Hyperparameters: This field can be use if you want to quickly override hyperparameters of the agent config, such as --hp={gamma:0.99,lr:0.001}.
- hp : the hyperparameters to override, as a dict string with no spaces.



# Results

### WandB
If wandb is activated, the results will be logged on wandb. For this you need to have a wandb account and to have specified your IDs in the settings.py file. 

### Tensorboard
If tensorboard is activated, the results will be logged on tensorboard in the logs/tb/ directory. You can start tensorboard with the following command:

```bash
tensorboard --logdir=logs/tb/
```

### Render
If n_render is specified, the agent will render the environment at each n_render episode.

### Logging
If the dump option is activated, the results will be logged in the logs/dumped_metrics folder (not available for now).

### Print
If the print option is activated, the results will be printed in the console.


# Agents

Agent are objects that can be trained on an environnement. They can be found in the rlearn/algorithms folder. The following agents are available:
- DQN
- PPO
- Random
- DDPG
- AC
- REINFORCE

For being used, an agent must implement the rlearn.algorithms.agent.Agent interface, i.e. act, remember and learn methods. Check the Agent class doctrings for more precision. It then must be added inside rlearn/implemented_agents.py.

For creating a new agent, you must :
- Create a new file for this agent, for example rlearn/algorithms/agents/my_agent.py, where other agents are.
- Implement the Agent interface, ie :
- Implement the act method, which returns an action given an observation.
- Implement the remember method, which stores the experience (observation, action, reward, next_observation, done) in the agent's memory.
- Implement the learn method, which trains the agent on the experience stored in the memory.
- Implement the get_state_types method, wichi returns which states the agent is able to handle (check docstring).
- In the init() method, you must call the init() method of Agent. This will set any attribute in the agent-cfg file as an attribute of the agent, define metrics and do other stuff.
- Add the agent to the rlearn/implemented_agents.py file









<!-- **For training an agent on a registered Gym environnement anywhere, run this command: (WIP)**

```bash
python -m rlearn train --agent <agent_name> --env <env_name>
``` -->


