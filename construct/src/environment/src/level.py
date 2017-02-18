import numpy as np
import matplotlib.pyplot as plt
import cairo
import math
#from noise import snoise2
import xml.etree.ElementTree
import tempfile
import os
from tqdm import tqdm
import uuid


def draw(objects, width, height):
    image = np.zeros((height, width, 4), dtype=np.uint8)
    ims = cairo.ImageSurface.create_for_data(image, cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(ims)
    for (x, y, z, r, color) in objects:
        cr.arc(x, y, r, -2 * math.pi, 0)  # Arc(cx, cy, radius, start_angle, stop_angle)

        if color == 2:
            cr.set_source_rgb(255, 0, 0)
        else:
            cr.set_source_rgb(0, 0, 0)

        cr.fill()
    return image


def is_valid_object(objects, obj, eps=0.0):
    points = objects[:, 0:2]
    x = np.sqrt(((points - obj[0:2])**2).sum(axis=1))
    x -= objects[:, 3]
    valid = x > (obj[3] + eps)

    return valid.all()


class UniformDistribution:
    def __init__(self, width, height):
        np.random.seed(uuid.uuid4().int % 100000)
        self.width, self.height = width, height

    def sample(self):
        return np.random.randint(0, self.width), np.random.randint(0, self.height)

def generate_world(num_objects):

    width, height = 20, 100
    spread_factor = 1
    gap = 1

    density_dist = UniformDistribution(width, height)

    sizes = [.25, .5, 2, 3, 8]
    size_p = [.10, .60, .20, .09, .01]
    attempts_per_object = 100

    goal_height = np.random.uniform(5, 20)
    objects = np.zeros([num_objects, 5])
    objects[0] = np.array([width / 2, 0, goal_height, 5, 0])  # Start
    objects[1] = np.array([width / 2, height, goal_height, 2, 1])  # End

    for i in range(2, num_objects):
        sampled_size = np.random.choice(sizes, p=size_p)
        for _ in range(attempts_per_object):
            x, y = density_dist.sample()
            z = 0
            obj_proposed = np.array([x, y, z, sampled_size, 2])
            if is_valid_object(objects, obj_proposed, eps=sampled_size + gap):
                objects[i] = obj_proposed
                break

    def objects_to_world(objects, base_file="basic.world"):
        sdf = xml.etree.ElementTree.ElementTree()
        sdf.parse(os.path.dirname(os.path.abspath(__file__)) + "/../worlds/" + base_file)
        world = sdf.find('world')

        new_tree_model = lambda: xml.etree.ElementTree.parse(os.path.dirname(os.path.abspath(__file__)) + "/../object_models/tree.sdf").getroot()
        new_goal_model = lambda: xml.etree.ElementTree.parse(os.path.dirname(os.path.abspath(__file__)) + "/../object_models/goal.sdf").getroot()

        for i, (x, y, z, r, type) in enumerate(objects):
            # if type == 1:
            #     obj = new_goal_model()

            h = np.random.uniform(1, goal_height * 3)

            if type == 2:
                obj = new_tree_model()
                obj.find('link/collision/geometry/cylinder/radius').text = str(r / spread_factor)
                obj.find('link/collision/geometry/cylinder/length').text = str(h)
                obj.set('name', "object_" + str(i) + "_type_" + str(type))
                obj.find('pose').text = str(x) + " " + str(y) + " " + str(h / 2) + " 0 0 0"
                obj.find('link/visual/geometry/cylinder/radius').text = str(r / spread_factor)
                obj.find('link/visual/geometry/cylinder/length').text = str(h)

                world.append(obj)

        _, filename = tempfile.mkstemp(suffix=".xml")
        with open(filename, mode='wb') as file_handler:
            sdf.write(file_handler)

        return filename

    map = draw(objects, width, height)
    objects[:, 0] = (objects[:, 0] - width / 2.0) / spread_factor
    objects[:, 1] = (objects[:, 1] - height / 2.0) / spread_factor

    return objects_to_world(objects), objects[0][0:3], objects[1][0:3], map

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    world_file, start, goal, map = generate_world(150)

    plt.imshow(map)
    plt.show()
