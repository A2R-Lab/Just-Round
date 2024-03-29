# Just-Round
Supplementary code for "Just Round: Quantized Observation Spaces Enable Memory Efficient Learning of Dynamic Locomotion"

## Installation
1. Install `stable_baselines3` from https://github.com/DLR-RM/stable-baselines3
2. Install `mujoco-py` from https://github.com/openai/mujoco-py
3. Run `python3 setup.py install` or `python3 setup.py develop`

## Example

### PPO
```python3
import gym
from stable_baselines3 import PPO

from just_round import QuantizedRolloutBuffer

env = gym.make("Walker2d-v3")

# For PPO, need to explicitly initialize rollout_buffer
model = PPO("MlpPolicy", env, verbose=1)
model.rollout_buffer = QuantizedRolloutBuffer(
    model.n_steps,
    model.observation_space,
    model.action_space,
    device=model.device,
    gamma=model.gamma,
    gae_lambda=model.gae_lambda,
    n_envs=model.n_envs,
    dec=1,
)
model.learn(total_timesteps=100000)
```

### SAC
```python3
import gym
from stable_baselines3 import SAC

from just_round import QuantizedReplayBuffer

env = gym.make("Walker2d-v3")

# For SAC, can make use of the replay_buffer_class initialization argument
model = SAC("MlpPolicy", env, verbose=1, replay_buffer_class=QuantizedReplayBuffer)
model.learn(total_timesteps=10000)
```
### Citing
To cite this work in your research, please use the following bibtex:
```
@inproceedings{grossman23JustRound,
  author = {Grossman, Lev and Plancher, Brian},
  title = {Just Round: Quantized Observation Spaces Enable Memory Efficient Learning of Dynamic Locomotion},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  address = {London, UK},
  month={May.},
  year = {2023}
}
```
