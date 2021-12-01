#!usr/bin/env python3

import os
import numpy as np
from stable_baselines3 import SAC
import kuka_env

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('outputDirName', 'output', 'output directory')
flags.DEFINE_string('savedModel', "./output/model.pth", 'trained model name')
flags.DEFINE_integer('playTimes', 1, 'Iteration times of play')


def test(argv):
    del argv

    model_path = FLAGS.savedModel
    assert os.path.exists(model_path)

    test_env = kuka_env.get_test_env(filename="kuka_grasp_sac", dir=os.path.join(".", FLAGS.outputDirName, "video"))
    model = SAC.load(path=model_path)

    for times in range(FLAGS.playTimes):
        print(f"Play {times + 1} ", end="")
        state = test_env.reset()
        while True:
            print(".", end="")
            action, _ = model.predict(state)
            state, _, done, info = test_env.step(action)
            if done:
                print(f"Grasp {'success' if info[0]['grasp_success'] else 'failed'}")
                break


if __name__ == '__main__':
    app.run(test)
