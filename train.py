#!/usr/bin/env python3

import os
import numpy as np

from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3 import SAC

import kuka_env

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('outputDir', 'output', 'output directory')
flags.DEFINE_string('savedModel', 'model.pth', 'trained model name')
flags.DEFINE_string('logName', "log", "name of tensorflow board log file")


def train(argv):
    del argv
    
    output = os.path.abspath(os.path.join(".", FLAGS.outputDir, ""))
    os.makedirs(output, exist_ok=True)
    tfboard_path = os.path.join(output, "tensorflow_board")
    output_model_path = os.path.join(output, FLAGS.savedModel)

    env = kuka_env.get_train_env(output)

    sac_policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(qf=[128, 64, 32], pi=[128, 64])
    )

    # create RL model
    controller = SAC('CnnPolicy', env, verbose=1, buffer_size=30000, batch_size=256,
                     policy_kwargs=sac_policy_kwargs, tensorboard_log=tfboard_path)

    # learn and save model
    controller.learn(total_timesteps=50000, log_interval=4, tb_log_name=FLAGS.logName)
    controller.save(path=output_model_path)


if __name__ == '__main__':
    app.run(train)
