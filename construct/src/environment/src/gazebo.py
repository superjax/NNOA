import subprocess
import os
import rospy
import atexit
from std_srvs.srv import Empty
from std_msgs.msg import Bool, Int16
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

        self.world_file, self.start, self.goal, self.map = generate_world(num_objects=2)

        # Something strange that's required
        self.goal[1] *= -1
        self.goal[2] *= -1

        self.namespace = '/' + str(supernamespace) + str(uuid.uuid4().get_hex().upper())

        max_step_size = self._get_step_size(self.world_file)
        update_rate = self._get_update_rate(self.path + '/../models/agent_models/' + agent + '.xacro')
        self.step_size = (1.0 / update_rate) / max_step_size * 2 - 1

        assert math.floor((1.0 / update_rate) / max_step_size) == (1.0 / update_rate) / max_step_size

    def run(self):
        arguments = [os.path.abspath(self.path + "/../launch/basic.launch"),
                     'x:=' + str(self.start[0]),
                     'y:=' + str(self.start[1]),
                     'z:=' + str(self.start[2]),
                     'yaw:=' + str(90),
                     'world_file:=' + self.world_file,
                     'render:=' + str(False),
                     'mav_name:=' + self.agent,
                     'verbose:=' + str(self.verbose),
                     'ns:=' + self.namespace,
                     'gzserver_port:=' + str(int(self.port))
                     ]

        self.process = _popen(["roslaunch"] + arguments, self.verbose, sleeptime=5)
        rospy.wait_for_service(self.namespace + '/gazebo/reset_simulation', timeout=5)

    def end(self):
        if self.process:
            # Hacky, but required.
            # You may think I should have used .terminate() or .kill() on self.process, but you would be wrong
            # roslaunch needs to catch the signal to pass it to it's spawned processes, so .kill() is off the table
            # and gzserver doesn't always catch the signal passed from roslaunch when using .terminate(), resulting
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
    def _get_update_rate(xacro_filename):
        t = xml.etree.ElementTree.parse(xacro_filename).getroot().find('./{http://ros.org/wiki/xacro}step_camera')
        assert t is not None, 'step_camera not found in ' + xacro_filename
        return int(t.attrib['frame_rate']) if t is not None else 25


class GazeboEnvironment:
    class Status(Enum):
        flying = 1
        crashed = 0
        error = 2
        timeout = 3
        success = 4

    class WorldManager:
        def __init__(self, agent_type, verbose, threads, supernamespace="PYTHONGAZEBO"):
            self.is_running = Value('b', True)
            self.worlds = Queue(2)
            self.total_worlds = Value('d', 0)
            self.verbose = verbose
            self.agent_type = agent_type
            self.current_world = None
            self.threads = threads
            self.supernamespace = supernamespace

            self.roscore = None

        def kill(self):
            self.is_running.value = False

        def start(self):
            self.roscore = _popen(['roscore'], self.verbose, sleeptime=1)
            rospy.set_param('/use_sim_time', True)
            rospy.init_node('pythonsim', anonymous=True, disable_signals=True)

            creation_threads = []
            for _ in range(self.threads):
                p = Process(target=self._world_thread,
                            args=[self.worlds, self.is_running, self.agent_type, self.total_worlds, self.verbose, self.supernamespace])
                p.daemon = True
                p.start()
                creation_threads.append(p)

            atexit.register(self._on_end, self.worlds, self.is_running, self.roscore, creation_threads)

            # Wait for worlds
            w = self.worlds.get(block=True)
            self.worlds.put(w)

        def connect(self):
            while self.is_running.value:
                try:
                    self.current_world = self.worlds.get(timeout=1)
                    return self.current_world
                except QueueEmpty as e:
                    pass

        @staticmethod
        def _world_thread(worlds, is_running, agent, total_worlds, verbose, supernamespace):
            miniworld = None
            while is_running.value:
                total_worlds.value += 1
                port_delta = random.randrange(100, 1000) + total_worlds.value
                miniworld = World(agent, 11345 + port_delta, verbose, supernamespace)

                try:
                    if is_running.value:
                        miniworld.run()

                    while is_running.value:
                        try:
                            worlds.put(miniworld, timeout=.1)
                            break
                        except QueueFull as e:
                            pass

                except KeyboardInterrupt:
                    is_running.value = False

                except rospy.ROSException as e:
                    pass

            # Worlds, created, but not in the queue that won't be cleaned up by _on_end
            if miniworld is not None:
                miniworld.end()

        def _on_end(self, worlds, is_running, roscore, creation_threads):
            with is_running.get_lock():
                is_running.value = False

            if self.current_world is not None:
                self.current_world.end()

            while not worlds.empty():
                world = worlds.get()
                world.end()
            worlds.close()
            worlds.join_thread()

            for p in creation_threads:
                p.terminate()
                p.join()

            rospy.signal_shutdown('')
            roscore.terminate()
            roscore.wait()

    def __init__(self):
        self.image_bridge = CvBridge()
        self.frame = 0
        self.max_frames = 1000

        self.verbose = False
        self.render = False

        self.current_world = None
        self.last_command = None
        self.listeners = []
        self.state = None
        self.frame_lock = Semaphore(1)
        self.flying_status = None
        self.odometry = None
        self.goal_radius = 2

        # TODO: read resolution from xacro
        # TODO: get bounds from James
        self._action_space = spaces.Box(low=-100, high=100, shape=(4,))
        self._observation_space = spaces.Tuple([spaces.Box(low=-100, high=100, shape=(7,)), spaces.Box(low=0, high=255, shape=(320, 240, 1))])

        self.agent_name = 'neo'

        self.world_manager = GazeboEnvironment.WorldManager(agent_type=self.agent_name, verbose=self.verbose, threads=1)
        self.world_manager.start()

        self.reset()

    def _connect(self):
        if self.current_world is not None:
            self.current_world.end()
            [l.unregister() for l in self.listeners]

        self.current_world = self.world_manager.connect()

        if self.current_world is not None:
            ns = self.current_world.namespace

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

            return True

        return False

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
                self.flying_status = self.Status.crashed

    def _on_odometry(self, message):
        self.odometry = message

    def _send_command(self, data):
        cmd = Command()
        cmd.mode = Command.MODE_XVEL_YVEL_YAWRATE_ALTITUDE
        cmd.x = data[0]
        cmd.y = data[1]
        cmd.z = data[2]
        cmd.F = data[3]
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
        return (normal * force).dot(np.array([1, 1, 1])) > 1

    def _reward_(self, before, after):
        if before['odometry'] and after['odometry']:
            then = np.array([before['odometry'].pose.pose.position.x, before['odometry'].pose.pose.position.y, before['odometry'].pose.pose.position.z])
            now = np.array([after['odometry'].pose.pose.position.x, after['odometry'].pose.pose.position.y, after['odometry'].pose.pose.position.z])
            return euclidean(then, self.current_world.goal) - euclidean(now, self.current_world.goal)
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

            if euclidean(position, self.current_world.goal) < self.goal_radius:
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

    @timeout_decorator.timeout(1)
    def _step(self):
        last = self.frame
        self.step_op.publish(self.current_world.step_size)

        # Infinite loop, timeout_decorator handles timeout
        # I couldn't get acquire's timeout argument to work when sending SIGINT with ctrl+c
        self.frame_lock.acquire(block=True)

        # Let the user know how many frames actually transpired between pauses
        return self.frame - last

    def step(self, command):
        self._send_command(command)
        before_state = self.state
        self._step()
        reward = self._reward_(before_state, self.state)
        terminal = self._terminal()

        return self._state_to_observation(self.state), reward, terminal, {}

    def render(self):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, attempt=0):
        if self._connect():
            self.unpause_sim_op()
            self.flying_status = self.Status.flying
            self.state = None
            self.odometry = None
            self._set_waypoint(self.current_world.goal)
            self.frame = 0
            self.last_command = None

            while self.frame == 0 and self.state is None:
                self._step()

            if self._terminal():
                self.reset()

            return self._state_to_observation(self.state)

        raise Exception