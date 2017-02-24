from construct import GazeboEnvironment
import cv2
from tqdm import tqdm

cv2.startWindowThread()
cv2.namedWindow("preview", cv2.CV_WINDOW_AUTOSIZE)

g = GazeboEnvironment(verbose=True)

for e in tqdm(range(10000)):
    g.reset()
    for i in tqdm(range(100)):
        action = g.action_space.sample()
        state, reward, terminal, _ = g.step([25.0, 0.0, 0.0, -10.0])
        cv2.imshow("preview", state[1])

        if terminal:
            g.reset()
            break