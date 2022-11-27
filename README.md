# **<project>**

# A Deep Reinforcement Learning framework : RLearn

RLearn is a framework for Deep Reinforcement Learning made by myself. I implements diverse RL algorithms and try to make them learn on various environnements. It is based on the [OpenAI Gym](https://gym.openai.com/) and [pytorch](https://pytorch.org/).

<p align="center">
  <img src="assets/myproject.png" alt="The CartPole environnement" width="60%"/>
</p>

# Table of Contents

-   [**Installation**](#installation)
-   [**Usage**](#usage)
-   [**Agents**](#agents)

# Installation

**Option 1 : Install the package from PyPI:**

```bash
pip install rlearn (WIP)
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
python run.py --algo <algo> --env <env>
```

Please note that your Gym environnement must be registered in order to be used.

**Fields:**

Required fields (all other are optional):
- algo : the name of your algorithm (e.g. DQN, PPO, random, ...). See the list of available algorithms below.
- env : the name of your environnement (e.g. CartPole-v0, LunarLander-v2, ...). The env must be registered as a Gym environnement.

Training fields:
- train : whether to train the agent or not.
- load : whether to load the agent or not.  (WIP)
- load_path : the path to the agent to load. By default, the most advanced (in terms of timesteps spend training on the env) agent will be loaded.

Configurations: The default configs are in configs/default/
- agent-cfg : the path to the config file of the agent. A default config file for the agent will be used if not specified.
- train-cfg : the path to the config file of the training. A default config file for the training will be used if not specified.

Feedback and logging:
- wandb : whether to use wandb or not. If true you must have specified your IDs in the settings.py file.
- tb : whether to use tensorboard or not.
- n_render : the frequency (one episode each n_render) of episodes to render.

Hyperparameters: This field can be use if you want to quickly override hyperparameters of the agent config, such as --hp={gamma:0.99,lr:0.001}.
- hp : the hyperparameters to override.




# Agents

Agent are objects that can be trained on an environnement. They can be found in the rlearn/algorithms folder. The following agents are available:
- DQN
- PPO
- Random
- DDPG
- AC
- REINFORCE

For being used, an agent must implement the rlearn.algorithms.agent.Agent interface, i.e. act, remember and learn methods. It then must be added inside rlearn/implemented_agents.py










<!-- **For training an agent on a registered Gym environnement anywhere, run this command: (WIP)**

```bash
python -m rlearn train --agent <agent_name> --env <env_name>
``` -->


