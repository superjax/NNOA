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


        # action = np.clip([.5 * (gx - cx)*math.cos(cyaw), .5 * (gy - cy)*math.sin(cyaw), 0, -10.0], -1, 1)
        action = [5, 0, 0, -10.0]

        print "cyaw:{:5f}, gyaw:{:5f}, cx:{:5f}, gx:{:5f}, cy:{:5f}, gy:{:5f}, action:{}".format(cyaw, gyaw, cx, gx, cy, gy, action)

        action[3] = -10

        state, reward, terminal, _ = g.step(action)

        if terminal:
            break