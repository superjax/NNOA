from construct import GazeboEnvironment
import cv2
from tqdm import tqdm

cv2.startWindowThread()
cv2.namedWindow("preview", cv2.CV_WINDOW_AUTOSIZE)

g = GazeboEnvironment()

for e in tqdm(range(1000)):
    for i in tqdm(range(1000)):
        action = g.action_space.sample()
        state, reward, terminal, _ = g.step(action)
        cv2.imshow("preview", state[1])

        if terminal:
            g.reset()
            break