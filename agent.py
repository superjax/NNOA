import numpy as np
from data_generation.models import Network
from construct import GazeboEnvironment
import cv2
from tqdm import tqdm
import imageio

g = GazeboEnvironment()

nn = Network()
nn.load_network("./.tfcheckpoints/3962000weighted_loss_bigger_batch_lr=0.000001.ckpt")

cv2.startWindowThread()
cv2.namedWindow("preview", cv2.CV_WINDOW_AUTOSIZE)

state = {'camera': np.zeros([240, 320, 1])}
data = np.zeros([10])

with imageio.get_writer('agent.gif', mode='I', fps=25, palettesize=16) as writer:
    for i in tqdm(range(2000)):
        inCam = state['camera']
        action = nn.get_output(np.array([inCam]), np.array([data]))

        writer.append_data(state['camera'])

        _, ins = nn.get_input([state['camera']], [data])
        state, reward, terminal = g.act(action)

        # print inCam.sum(), data.sum(), action, ins

        pose = state['odometry'].pose.pose
        data[0:3] = g.current_world.goal
        data[3:6] = [pose.position.x, pose.position.y, pose.position.z]
        data[6:] = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        cv2.imshow("preview", state['camera'])

        if terminal:
           g.reset()

