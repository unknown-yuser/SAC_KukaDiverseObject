#!usr/bin/env python3

import os
import numpy as np
from stable_baselines3 import SAC
import kuka_env

model_path = "./output/model.pth"
assert os.path.exists(model_path)

test_env = kuka_env.get_test_env(filename="sac_result", dir="./output/video")
model = SAC.load(path=model_path)

# Try 10 times
for count in range(10):
    print(f"Play {count + 1} ", end="")
    state = test_env.reset()
    while True:
        print(".", end="")
        action, _ = model.predict(state)
        state, _, done, info = test_env.step(action)
        if done:
            print(f"Grasp {'success' if info[0]['grasp_success'] else 'failed'}")
            break
