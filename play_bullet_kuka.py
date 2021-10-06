import os
import gym
from gym.wrappers.monitor import video_recorder
from gym import Wrapper
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np

from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3 import SAC

import time
import threading


def capture_runner(record_video_env):
    while not record_video_env.step_done:
        record_video_env.video_recorder.capture_frame()
        time.sleep(1.0 / 30.0)


class KukaRecordVideo(Wrapper):
    def __init__(
            self,
            env,
            name_prefix,
            video_folder,
    ):
        super(KukaRecordVideo, self).__init__(env)
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)
        self.name_prefix = name_prefix
        self.recording = False
        self.step_id = 0
        self.episode_id = 0

        self.capture_runner_thread = None
        self.step_done = False

    def reset(self, **kwargs):
        obs = super(KukaRecordVideo, self).reset(**kwargs)
        if not self.recording:
            self.start_video_recorder()
        return obs

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        )
        self.video_recorder.capture_frame()
        self.recording = True

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False

    def _before_step(self):
        self.step_done = False
        self.capture_runner_thread = threading.Thread(
            target=capture_runner,
            args=(self,))
        self.capture_runner_thread.start()

    def _after_step(self):
        self.step_done = True
        self.capture_runner_thread.join()

    def step(self, action):
        self._before_step()
        obs, reward, done, info = super(KukaRecordVideo, self).step(action)
        self._after_step()

        self.step_id += 1
        if done:
            self.episode_id += 1

        return obs, reward, done, info


class ImageObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(ImageObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=env.observation_space.shape, dtype=np.uint8)


def render_video(model, try_count=1, prefix='', video_folder="kuka_video"):
    """
    :param model:
    :param try_count:
    :param prefix:
    :param video_folder:
    :return:
    """
    target_env = KukaDiverseObjectEnv(maxSteps=20, isDiscrete=False, renders=True, removeHeightHack=True, isTest=True)
    target_env = KukaRecordVideo(env=target_env, name_prefix=prefix, video_folder=video_folder)

    cnt = 0
    while cnt < try_count:
        obs = target_env.reset()
        for _ in range(500):
            action, _ = model.predict(obs)
            obs, _, done, _ = target_env.step(action)
            if done:
                cnt += 1
                break


result_path = "./kuka_result/"
result_path = os.path.abspath(result_path)
os.makedirs(result_path, exist_ok=True)

video_record_path = os.path.join(result_path, "video/")

env = KukaDiverseObjectEnv(maxSteps=20, isDiscrete=False, renders=False, removeHeightHack=True)
env = ImageObservationWrapper(env)
env = Monitor(env, result_path)
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(qf=[128, 64, 32], pi=[128, 64])
)

# create RL model
controller = SAC('CnnPolicy', env, verbose=1, buffer_size=30000, batch_size=256, policy_kwargs=policy_kwargs,
                 tensorboard_log=os.path.join(result_path, "tb_log"))

controller.learn(total_timesteps=100000, log_interval=4, tb_log_name='kuka_sac')
# controller.save(path=os.path.join(result_path, "kuka_sac.pkl"))

# controller = SAC.load(path=os.path.join(result_path, "kuka_sac.pkl"))
render_video(controller, try_count=10, prefix="kuka_res", video_folder=os.path.join(result_path, video_record_path))
