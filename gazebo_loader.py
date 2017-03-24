from tqdm import tqdm
import timeout_decorator
import cPickle as pkl
import cv2
import imageio
import uuid
import os
from construct import GazeboEnvironment
import numpy as np
import math
import tf.transformations as tft

#with imageio.get_writer('video.gif', mode='I', fps=50, palettesize=16) as writer:
# writer.append_data(g.state['camera'])
# cv2.startWindowThread()
# cv2.namedWindow("preview", cv2.CV_WINDOW_AUTOSIZE)

# cv2.destroyAllWindows()

# import pandas as pd
# index = pd.read_csv('/mnt/pccfs/data/nnoa/75AFB88781054B3983768791B316C16C/index.csv')
# print np.load(index.iloc[0, 0])

g = GazeboEnvironment()

@timeout_decorator.timeout(5)
def wait(g):
    while g.last_command is None:
        pass
    lc = g.last_command
    g.last_command = None
    return lc


uid = str(uuid.uuid4().get_hex().upper())
base_path = "/mnt/pccfs/not_backed_up/data/nnoa/{}".format(uid)
os.makedirs(base_path)
ep = 0

target_episodes = 100
data_bar = tqdm(total=target_episodes)

with open(base_path + "/index.csv", mode='a') as idx_file:
    while ep < target_episodes:
        state = g.reset()
        episode = []
        episode_bar = tqdm()
        try:
            while not g._terminal():
                episode_bar.update()

                gx, gy, gz = g.world_manager.world.goal
                cx, cy, cz = state[0][:3]
                qx, qy, qz, qw = state[0][3:7]
                r, p, cyaw = tft.euler_from_quaternion(state[0][3:7])
                gyaw = math.atan2(gy - cy, gx - cx)

                action = np.clip([.5 * (gy - cy) * math.sin(cyaw), .5 * (gx - cx) * math.cos(cyaw), gyaw - cyaw, -10.0], -10, 10)

                state, reward, terminal, _ = g.step(action)
                lc = wait(g)

                episode.append((g.state, lc))

            if g._terminal_status() == GazeboEnvironment.Status.success:
                data_bar.update()
                ep += 1
                episode_path = base_path + "/{}".format(ep)
                os.makedirs(episode_path)

                for current_index, (state, command) in tqdm(enumerate(episode), total=len(episode)):
                    file_path = "/{0}/{1:04d}.camera.npy".format(ep, current_index)
                    state_filename = base_path + file_path

                    with open(state_filename, mode='wb') as camera_file:
                        np.save(camera_file, state['camera'])

                    pose = state['odometry'].pose.pose
                    goal = [g.current_world.goal[0], g.current_world.goal[1], g.current_world.goal[2]]
                    command = [command.x, command.y, command.z, command.F]
                    position = [pose.position.x, pose.position.y, pose.position.z]
                    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    idx_file.writelines(["{},({}),({}),({}),({})\n".format(file_path,
                                                                      ",".join(map(str, command)),
                                                                      ",".join(map(str, goal)),
                                                                      ",".join(map(str, position)),
                                                                      ",".join(map(str, orientation)))])

                idx_file.flush()

        except timeout_decorator.TimeoutError as e:
            pass

        episode_bar.close()


cv2.destroyAllWindows()