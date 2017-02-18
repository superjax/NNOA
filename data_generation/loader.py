import os
import numpy as np
import pandas as pd

# Just loading one at the moment
DATA_ROOT = "/mnt/pccfs/data/nnoa"
DIRS = ["47BF54B3DEE049FDB6E986C495E39224", "A5020A3F9D3E4B1BA4184B5A314C0B89", "C2E63E0DDD1A4FB49AE66EF83FFF733B", "C70BF1E6D1B34E0F8F20634466DE458E", "EAD17AAE66D94084880FE5996E43F576"]

class DataNode():
    def __init__(self, image=None, command=None, goal=None, position=None, orientation=None):
        self._Image = image
        self._Command = command
        self._Goal    = goal
        self._Position = position
        self._Orientation = orientation

    def set_image(self, img):
        self._Image = img
    
    def set_command(self, cmd):
        self._Command = cmd

    def set_goal(self, goal):
        self._Goal = goal
    
    def set_position(self, pos):
        self._Position = pos
    
    def set_orientation(self, orientation):
        self._Orientation = orientation
    
    def load_data(self):
        img = np.load(self._Image)
        data = np.asarray(self._Goal + self._Position + self._Orientation)
        com = np.asarray(self._Command)

        return img, data, com
    
    def __str__(self):
        return self._Image + ", " + str(self._Command) + ", " + str(self._Goal) + ", " + str(self._Position) + ", " + str(self._Orientation)


def load_data(num_threads = 12):
    print "Loading data"
    data = []

    for d in DIRS:
        path = os.path.join(DATA_ROOT, d)
        pd_data = pd.read_csv(os.path.join(path, "index.csv"))
        all_data = pd_data.as_matrix()

        for i, x in enumerate(all_data):
            img = os.path.join(path, all_data[i][0][1:])
            command = (float(all_data[i][1][1:]), float(all_data[i][2]), float(all_data[i][3]), float(all_data[i][4][:-1]))
            goal = (float(all_data[i][5][1:]), float(all_data[i][6]), float(all_data[i][7][:-1]))
            pos = (float(all_data[i][8][1:]), float(all_data[i][9]), float(all_data[i][10][:-1]))
            rot = (float(all_data[i][11][1:]), float(all_data[i][12]), float(all_data[i][13]), float(all_data[i][14][:-1]))

            data.append( DataNode(img, command, goal, pos, rot) )
    
    print "Loaded data!"
    return data