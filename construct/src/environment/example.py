from construct import GazeboEnvironment
from tqdm import tqdm
import numpy as np
import math
import tf.transformations as tft

g = GazeboEnvironment(verbose=True)

for e in tqdm(range(10000)):
    state = g.reset()
    for i in (range(1000)):

        gx, gy, gz = g.world_manager.world.goal
        cx, cy, cz = state[0][:3]
        qx, qy, qz, qw = state[0][3:7]
        r, p, cyaw = tft.euler_from_quaternion(state[0][3:7])
        gyaw = math.atan2(gy - cy, gx - cx)

        action = np.clip([.5 * (gy - cy) * math.sin(cyaw), .5 * (gx - cx) * math.cos(cyaw), gyaw - cyaw, -10.0], -10, 10)

        state, reward, terminal, _ = g.step(action)

        if terminal:
            break