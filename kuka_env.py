#!usr/bin/env python3

import os
import threading
import numpy as np

import gym
from gym import Wrapper
from gym.wrappers.monitor import video_recorder
import pybullet_envs

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv


class AssignTypeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(AssignTypeWrapper, self).__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.uint8)


class KukaVideoRecorder(Wrapper):
    def __init__(self, env, filename, video_folder):
        super(KukaVideoRecorder, self).__init__(env)
        self.recording = False
        self.in_playing = False

        os.makedirs(video_folder, exist_ok=True)
        self.file_path = os.path.join(video_folder, filename)
        self.video_recorder = video_recorder.VideoRecorder(env=self.env, base_path=self.file_path)

        def snapshot_worker(recorder):
            while recorder.recording:
                if recorder.in_playing:
                    recorder.video_recorder.capture_frame()

        self.capture_runner_thread = threading.Thread(target=snapshot_worker, args=(self,), daemon=True)

    def reset(self, **kwargs):
        self.in_playing = False
        observation = self.env.reset(**kwargs)
        self.in_playing = True

        if not self.recording:
            self.recording = True
            self.capture_runner_thread.start()

        return observation

    def __del__(self):
        if self.recording:
            self.recording = False
            self.video_recorder.close()
            self.capture_runner_thread.join()


def get_train_env(filename: str):
    env = gym.make("KukaDiverseObjectGrasping-v0", maxSteps=20, isDiscrete=False, renders=False, removeHeightHack=True)
    env = AssignTypeWrapper(env)
    env = Monitor(env, filename=filename)
    env = VecTransposeImage(DummyVecEnv([lambda: env]))
    return env


def get_test_env(filename: str, dir: str):
    env = gym.make("KukaDiverseObjectGrasping-v0", maxSteps=20, isDiscrete=False, renders=True, removeHeightHack=True, isTest=True)
    env = AssignTypeWrapper(env)
    env = KukaVideoRecorder(env=env, filename=filename, video_folder=dir)
    env = VecTransposeImage(DummyVecEnv([lambda: env]))
    return env
