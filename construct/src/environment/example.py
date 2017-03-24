from construct import GazeboEnvironment
from tqdm import tqdm
import numpy as np
import math


g = GazeboEnvironment(verbose=True)

for e in tqdm(range(10000)):
    state = g.reset()
    for i in (range(1000)):
        action = [5, 0, 0, -10.0]
        state, reward, terminal, _ = g.step(action)

        if terminal:
            break