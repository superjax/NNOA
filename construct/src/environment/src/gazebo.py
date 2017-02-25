import subprocess
import os
import rospy
import atexit
from std_srvs.srv import Empty
from std_msgs.msg import Bool, Int16, String
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ContactsState
import time
import numpy as np
import math
import xml.etree.ElementTree
from level import generate_world
import timeout_decorator
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from fcu_common.msg import Command
from enum import Enum
import uuid
from multiprocessing import Queue, Process, Value, Semaphore
from Queue import Empty as QueueEmpty, Full as QueueFull
import os
import random
from gym import spaces
import signal

def _popen(command, verbose=True, sleeptime=1):
    process = subprocess.Popen(command,
                               stdout=open(os.devnull, 'w') if not verbose else None,
                               stderr=open(os.devnull, 'w') if not verbose else None)
    time.sleep(sleeptime)
    return process


class World:
    def __init__(self, agent, port, verbose, supernamespace):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose
        self.agent = agent
        self.port = port
        self.process = None

        self.world_file, self.start, self.goal, self.map = generate_world(num_objects=50)

        # There is an annoying flip that happens for the goal altitude
        self.goal[1] *= -1
        self.goal[2] *= -1

        self.namespace = '/' + str(supernamespace) + str(uuid.uuid4().get_hex().upper())

        max_step_size = self._get_step_size(self.world_file)
        update_rate = self._get_update_rate(agent)
        self.step_size = (1.0 / update_rate) / max_step_size * 2 - 1

        assert math.floor((1.0 / update_rate) / max_step_size) == (1.0 / update_rate) / max_step_size

    def run(self, pause_process=True):
        arguments = [os.path.abspath(self.path + "/../launch/basic.launch"),
                     'x:=' + str(self.start[0]),
                     'y:=' + str(self.start[1]),
                     'z:=' + str(self.start[2]),
                     'world_file:=' + str(self.world_file),
                     'yaw:=' + str(np.pi / 2.0),
                     'render:=' + str(False),
                     'mav_name:=' + self.agent,
                     'verbose:=' + str(self.verbose),
                     'ns:=' + self.namespace,
                     'gzserver_port:=' + str(int(self.port))
                     ]

        self.reset = rospy.ServiceProxy(self.namespace + '/gazebo/reset_simulation', Empty, persistent=False)
        self.randomize_obstacles = rospy.ServiceProxy(self.namespace + '/gazebo/randomize_obstacles', Empty, persistent=False)
        self.process = _popen(["roslaunch", '--screen'] + arguments, self.verbose, sleeptime=3)
        self.reset.wait_for_service(10)

    def end(self):
        if self.process:
            # Hacky, but required.
            # You may think I should have used .terminate() or .kill() on self.process, but you would be wrong
            # roslaunch needs to catch the signal to pass it to it's spawned processes, so .kill() is off the table
            # and gzserver doesn't catch the signal passed from roslaunch when using .terminate(), resulting
            # in processes that continue to run, and result in linearly increasing cpu/mem consumption and then hang
            # on world_creation_thread.wait() when the main program terminates since the creation threads can't terminate
            # UPDATE: I thought I could remove this, but I was wrong.
            _popen(['pkill', '-f', '-9', self.namespace[1:]], verbose=True, sleeptime=0)

    @staticmethod
    def _get_step_size(world_filename):
        t = xml.etree.ElementTree.parse(world_filename).getroot().find('world/physics/max_step_size')
        assert t is not None, 'max_step_size not found in ' + world_filename
        return float(t.text) if t is not None else 0.01

    @staticmethod
    def _get_resolution(agent):
        xacro_filename = os.path.dirname(os.path.abspath(__file__)) + '/../models/agent_models/' + agent + '.xacro'
        t = xml.etree.ElementTree.parse(xacro_filename).getroot().find('./{http://ros.org/wiki/xacro}step_camera')
        # assert t is not None, 'step_camera not found in ' + xacro_filename
        return (int(t.attrib['width']), int(t.attrib['height'])) if t is not None else (320, 250)

    @staticmethod
    def _get_update_rate(agent):
        xacro_filename = os.path.dirname(os.path.abspath(__file__)) + '/../models/agent_models/' + agent + '.xacro'
        t = xml.etree.ElementTree.parse(xacro_filename).getroot().find('./{http://ros.org/wiki/xacro}step_camera')
        # assert t is not None, 'step_camera not found in ' + xacro_filename
        return int(t.attrib['frame_rate']) if t is not None else 25


class GazeboEnvironment:
    class Status(Enum):
        flying = 1
        crashed = 0
        error = 2
        timeout = 3
        success = 4

    class SingleWorldManager:
        def __init__(self, agent_type, verbose, threads, supernamespace="PYGZ"):
            self.roscore = _popen(['rosmaster', '--core'], verbose, sleeptime=3)
            rospy.set_param('/use_sim_time', True)
            rospy.init_node('pythonsim', anonymous=True, disable_signals=True)

            self.world = World(agent_type, 11345, verbose, supernamespace)
            self.world.run(pause_process=False)

            atexit.register(self.kill)

        def kill(self):
            self.world.end()
            rospy.signal_shutdown('')
            self.roscore.terminate()
            self.roscore.wait()

    def __init__(self, verbose=False):
        self.image_bridge = CvBridge()
        self.frame = 0
        self.max_frames = 1000

        self.verbose = verbose
        self.render = False

        self.current_world = None
        self.last_command = None
        self.listeners = []
        self.state = None
        self.frame_lock = Semaphore(1)
        self.flying_status = None
        self.odometry = None
        self.goal_radius = 2

        self.last_frame = 0

        self.agent_name = 'neo'

        bounds = np.array([2*np.pi, 2*np.pi, 10, 1])
        self.action_space = spaces.Box(low=-bounds, high=bounds)
        self.resolution = World._get_resolution(self.agent_name)
        self.observation_space = spaces.Tuple([spaces.Box(low=-100, high=100, shape=(7,)),
                                                spaces.Box(low=0, high=255, shape=(self.resolution[0], self.resolution[1], 1))])

        self.world_manager = GazeboEnvironment.SingleWorldManager(agent_type=self.agent_name, verbose=self.verbose, threads=5)
        ns = self.world_manager.world.namespace

        self.listeners = [
            rospy.Subscriber(ns + '/' + self.agent_name + '/camera/rgb', Image, self._on_cameraframe, queue_size=1),
            rospy.Subscriber(ns + '/' + self.agent_name + '/ground_truth/odometry', Odometry, self._on_odometry, queue_size=1),
            rospy.Subscriber(ns + '/' + self.agent_name + '/contact', ContactsState, self._on_contact, queue_size=1),
            rospy.Subscriber(ns + '/' + self.agent_name + '/high_level_command', Command, self._on_command, queue_size=1)
        ]

        self.unpause_sim_op = rospy.ServiceProxy(ns + '/gazebo/unpause_physics', Empty)
        self.step_op = rospy.Publisher(ns + '/gazebo/step', Int16, queue_size=5)
        self.waypoint_op = rospy.Publisher(ns + '/' + self.agent_name + '/waypoint', Command, queue_size=5, latch=True)
        self.command_op = rospy.Publisher(ns + '/' + self.agent_name + '/high_level_command', Command, queue_size=5, latch=True)

        self.reset()

    def _on_cameraframe(self, message):
        self.state = {'camera': np.asarray(self.image_bridge.imgmsg_to_cv2(message, "mono8")),
                      'odometry': self.odometry}

        self.frame_lock.release()
        self.frame += 1

    def _on_command(self, message):
        self.last_command = message

    def _on_contact(self, message):
        for collision in message.states:
            if self._is_deadly_collision(collision):
                self.frame_lock.release()
                self.flying_status = self.Status.crashed

    def _on_odometry(self, message):
        self.odometry = message

    def _send_command(self, data):
        cmd = Command()
        cmd.mode = Command.MODE_XVEL_YVEL_YAWRATE_ALTITUDE
        cmd.x, cmd.y, cmd.z, cmd.F = data
        self.command_op.publish(cmd)

    def _set_waypoint(self, position):
        cmd = Command()
        cmd.x, cmd.y, cmd.z, cmd.F = position[0], position[1], 0, position[2]
        cmd.mode = Command.MODE_XPOS_YPOS_YAW_ALTITUDE 
        self.waypoint_op.publish(cmd)

    @staticmethod
    def _is_deadly_collision(collision):
        normal = np.abs(np.array([collision.contact_normals[0].x, collision.contact_normals[0].y, collision.contact_normals[0].z]))
        force = np.abs(np.array([collision.total_wrench.force.x, collision.total_wrench.force.y, collision.total_wrench.force.z]))
        return (normal * force).sum() > 1

    def _reward_(self, before, after):
        if before['odometry'] and after['odometry']:
            then = np.array([before['odometry'].pose.pose.position.x, before['odometry'].pose.pose.position.y, before['odometry'].pose.pose.position.z])
            now = np.array([after['odometry'].pose.pose.position.x, after['odometry'].pose.pose.position.y, after['odometry'].pose.pose.position.z])
            return euclidean(then, self.world_manager.world.goal) - euclidean(now, self.world_manager.world.goal)
        return 0

    def _terminal_status(self):
        status = self.flying_status
        if self.state['odometry']:
            p = self.state['odometry'].pose.pose.position
            position = np.array([p.x, p.y, p.z])

            if euclidean(position, np.zeros(3)) == 0 or euclidean(position, np.zeros(3)) > 1000:
                status = self.Status.error

            if self.frame > self.max_frames:
                status = self.Status.timeout

            if euclidean(position, self.world_manager.world.goal) < self.goal_radius:
                status = self.Status.success

        return status

    def _terminal(self):
        return self._terminal_status() != self.Status.flying

    def _state_to_observation(self, state):
        if state['odometry']:
            pose = state['odometry'].pose.pose
            odometry = [pose.position.x, pose.position.y, pose.position.z] + \
                       [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            return (odometry, state['camera'])
        return (np.zeros([7,]), state['camera'])

    @timeout_decorator.timeout(1.0)
    def _step(self):
        last = self.frame
        self.step_op.publish(self.world_manager.world.step_size)

        # Infinite loop, timeout_decorator handles timeout
        # I couldn't get acquire's timeout argument to work when sending SIGINT with ctrl+c
        self.frame_lock.acquire(block=True, timeout=0.80)
        # while last == self.frame:
        #     pass

        # Let the user know how many frames actually transpired between pauses
        return self.frame - last

    def step(self, command):
        self._send_command(command)
        before_state = self.state

        for attempt in range(3):
            try:
                self._step()
                reward = self._reward_(before_state, self.state)
                terminal = self._terminal()
                break
            except timeout_decorator.TimeoutError as e:
                terminal, reward = True, 0

        return self._state_to_observation(self.state), reward, terminal, {}

    def render(self):
        pass

    def reset(self):
        for attempt in range(3):
            try:
                self.world_manager.world.randomize_obstacles()
                self.world_manager.world.reset()
                self.unpause_sim_op()
                self.flying_status = self.Status.flying
                self.state = None
                self.odometry = None
                self.frame = 0
                self.last_command = None

                while self.frame == 0 and self.state is None:
                    self._step()

                if self._terminal():
                    self.reset()

                return self._state_to_observation(self.state)
            except timeout_decorator.TimeoutError:
                print 'Timeout Error'
                pass

        self.world_manager.kill()
        raise Exception('Consecutive timeouts experienced')