import os
import numpy as np
from random import randint
from tqdm import tqdm

import loader as ld
from models import Network

# CONSTANTS
#---------------------------------------------#
IMG_HEIGHT = 240
IMG_WIDTH = 320
IMG_CHANNELS = 1
N_IN_DATA = 10
N_OUT_DATA = 4
BATCH_SIZE = 80
USE_WEIGHTED_LOSS = False

SAVE_NETWORK = True
LOAD_NETWORK = False
SAVE_LOCATION = ".tfcheckpoints"
LOAD_LOCATION = "./.tfcheckpoints/4000scaled_loss.ckpt"
CHECKPOINT_END = "bigger_batch_lr=0.000001.ckpt"
#---------------------------------------------#
# Helper Functions
#---------------------------------------------#
def get_batch(in_data, batch_size):
    batch_imgs = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], np.float32)
    batch_data = np.empty([batch_size, N_IN_DATA], np.float32)
    batch_desi = np.empty([batch_size, N_OUT_DATA], np.float32)
    for i in range(batch_size):
        x = randint(1, len(in_data)) - 1
        batch_imgs[i], batch_data[i], batch_desi[i] = in_data[x].load_data()
    return batch_imgs, batch_data, batch_desi

#---------------------------------------------#
# Running the Network
#---------------------------------------------#

# Load the network and the data
data = ld.load_data()
nn = Network()
if LOAD_NETWORK:
    nn.load_network(LOAD_LOCATION)

# Main loop
for i in tqdm(range(4000, 10000000)):
    # Generate the batch and train
    img_batch, data_batch, desired_batch = get_batch(data, BATCH_SIZE)
    loss = nn.train(img_batch, data_batch, desired_batch, USE_WEIGHTED_LOSS)

    # Print the loss
    if i % 20 == 0:
        print i, loss, CHECKPOINT_END

    # Save the network
    if SAVE_NETWORK and (i + 1) % 1000 == 0:
        nn.save_network(os.path.join(SAVE_LOCATION, str(i + 1) + CHECKPOINT_END))
