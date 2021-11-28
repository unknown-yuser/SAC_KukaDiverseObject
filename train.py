#!/usr/bin/env python3

import os

import numpy as np
import gym
import pybullet_envs

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3 import SAC

import time


class AssignTypeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(AssignTypeWrapper, self).__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.uint8)


output = os.path.abspath("./output/")
os.makedirs(output, exist_ok=True)
tfboard_path = os.path.join(output, "sac_learn_log")
output_model_path = os.path.join(output, "model.pth")

env = gym.make("KukaDiverseObjectGrasping-v0", maxSteps=20, isDiscrete=False, renders=False, removeHeightHack=True)
env = AssignTypeWrapper(env)
env = Monitor(env, output)
env = VecTransposeImage(DummyVecEnv([lambda: env]))

sac_policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(qf=[128, 64, 32], pi=[128, 64])
)

# create RL model
controller = SAC('CnnPolicy', env, verbose=1, buffer_size=30000, batch_size=256,
                 policy_kwargs=sac_policy_kwargs, tensorboard_log=tfboard_path)

# learn and save model
controller.learn(total_timesteps=100000, log_interval=4, tb_log_name='kuka_sac_log')
controller.save(path=output_model_path)
