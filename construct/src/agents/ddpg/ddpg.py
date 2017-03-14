from construct import GazeboEnvironment
from memory import Memory
from tqdm import tqdm
import tensorflow as tf
from collections import deque
import tensorflow.contrib.slim as slim
import numpy as np
from scipy import stats

env = GazeboEnvironment(verbose=False)

TAU = 0.01
GAMMA = .99
ACTOR_LR = 0.001
CRITIC_LR = 0.001
ACTOR_L2_WEIGHT_DECAY = 0.00
CRITIC_L2_WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
MAX_EPISODE_LENGTH = 1000
ITERATIONS_BEFORE_TRAINING = BATCH_SIZE + 1

ACTION_DIM = env.action_space.shape[0]

assert len(env.action_space.shape) == 1


def fanin_init(layer):
    fanin = layer.get_shape().as_list()[1]
    v = 1.0 / np.sqrt(fanin)
    return tf.random_uniform_initializer(minval=-v, maxval=v)

def bounded_constraint(forward_op, min=-1.0, max=1.0):
    gradient_op_name = "BoundedConstraint-" + str(np.random.randint(1000, 9999))

    @tf.RegisterGradient(gradient_op_name)
    def plus_minus_one_gradient(op, grad):
        # Inverting Gradients for Bounded Control - https://www.cs.utexas.edu/~AustinVilla/papers/ICLR16-hausknecht.pdf
        return [grad * tf.select(grad < 0.0, tf.sign((max - op.outputs[0]) / (max-min)), tf.sign((op.outputs[0] - min) / (max-min)))]

    with forward_op.graph.gradient_override_map({"Identity": gradient_op_name}):
        return tf.identity(forward_op)


class Noise:
    def __init__(self, ):
        self.state = np.zeros(ACTION_DIM)

    def josh(self, mu, sigma):
        return stats.truncnorm.rvs((-1 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma, size=len(self.state))

    def reflected_ou(self, mean, sigma=.1, theta=.1, min=0, max=1):
        theta, sigma, min, max = map(np.array, [theta, sigma, min, max])
        mean = np.clip(mean, min, max)
        mu = self.state + -theta * (self.state - mean)
        self.state = stats.truncnorm.rvs((min - mu) / sigma, (max - mu) / sigma, loc=mu, scale=sigma, size=len(self.state))
        return self.state.copy()

    def ou(self, mean, theta=.2, sigma=.15):
        sigma, theta = np.array(sigma), np.array(theta)
        self.state += -theta * (self.state - mean) - sigma * np.random.randn(len(self.state))
        return self.state.copy()


def actor_network(states, outer_scope, reuse=False):
    with tf.variable_scope(outer_scope + '/actor', reuse=reuse):
        uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        state = states[1]
        net = slim.conv2d(inputs=state, num_outputs=32, kernel_size=5, weights_initializer=uniform_random,
                          biases_initializer=uniform_random)
        net = slim.conv2d(inputs=net, num_outputs=30, kernel_size=1, stride=5, padding="VALID",
                          weights_initializer=uniform_random, biases_initializer=uniform_random)
        net = tf.concat(1, [tf.contrib.layers.flatten(l) for l in [states[0], net]])

        net = slim.fully_connected(net, 100, weights_initializer=uniform_random, biases_initializer=uniform_random)
        net = slim.fully_connected(net, 100, weights_initializer=uniform_random, biases_initializer=uniform_random)
        output = slim.fully_connected(net, ACTION_DIM, weights_initializer=uniform_random,
                                      biases_initializer=uniform_random, activation_fn=None)

        output = bounded_constraint(output, min=-1.0, max=1.0)

    return output


def critic_network(states, action, outer_scope, reuse=False):
    with tf.variable_scope(outer_scope + '/critic', reuse=reuse):
        uniform_random = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        state = states[1]
        net = slim.conv2d(inputs=state, num_outputs=32, kernel_size=5, weights_initializer=uniform_random,
                          biases_initializer=uniform_random)
        net = slim.conv2d(inputs=net, num_outputs=30, kernel_size=1, stride=5, padding="VALID",
                          weights_initializer=uniform_random, biases_initializer=uniform_random)
        net = tf.concat(1, [tf.contrib.layers.flatten(l) for l in [net, action, states[0]]])
        net = slim.fully_connected(net, 100, weights_initializer=uniform_random, biases_initializer=uniform_random)
        net = slim.fully_connected(net, 100, weights_initializer=uniform_random, biases_initializer=uniform_random)
        net = slim.fully_connected(net, 1, weights_initializer=uniform_random, biases_initializer=uniform_random,
                                   activation_fn=None)

        return tf.squeeze(net, [1])


# state_placeholders = [tf.placeholder(tf.float32, [None, 1] + list(space.shape), 'state'+str(i)) for i, space in enumerate(env.observation_space.spaces)]
# next_state_placeholder = [tf.placeholder(tf.float32, [None, 1] + list(space.shape), 'nextstate'+str(i)) for i, space in enumerate(env.observation_space.spaces)]

odometry_placeholder = tf.placeholder(tf.float32, [None, 1] + list(env.observation_space.spaces[0].shape), 'odometry')
camera_placeholder = tf.placeholder(tf.uint8, [None, 1] + list(env.observation_space.spaces[1].shape), 'camera')

next_odometry_placeholder = tf.placeholder(tf.float32, [None, 1] + list(env.observation_space.spaces[0].shape), 'next_odometry')
next_camera_placeholder = tf.placeholder(tf.uint8, [None, 1] + list(env.observation_space.spaces[1].shape), 'next_camera')

state_placeholders = [odometry_placeholder, tf.cast(camera_placeholder, tf.float32) / 255.0]
next_state_placeholders = [next_odometry_placeholder, tf.cast(next_camera_placeholder, tf.float32) / 255.0]

action_placeholder = tf.placeholder(tf.float32, [None, ACTION_DIM], 'action')
reward_placeholder = tf.placeholder(tf.float32, [None], 'reward')
done_placeholder = tf.placeholder(tf.bool, [None], 'done')

train_actor_output = actor_network(state_placeholders, outer_scope='train_network')
train_actor_next_output = actor_network(next_state_placeholders, outer_scope='train_network', reuse=True)
target_actor_output = actor_network(state_placeholders, outer_scope='target_network')
target_actor_next_output = actor_network(next_state_placeholders, outer_scope='target_network', reuse=True)

target_critic_next_output = critic_network(next_state_placeholders, target_actor_next_output, outer_scope='target_network')
train_critic_current_action = critic_network(state_placeholders, train_actor_output, outer_scope='train_network')
train_critic_placeholder_action = critic_network(state_placeholders, action_placeholder, outer_scope='train_network', reuse=True)

train_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/actor')
target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/actor')
train_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/critic')
target_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network/critic')

print 'Total Actor Parameters:', sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list()) for l in e]) for e in [train_actor_vars]])
print 'Total Critic Parameters:', sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list()) for l in e]) for e in [train_critic_vars]])

with tf.name_scope('actor_loss'):
    weight_decay_actor = tf.add_n([ACTOR_L2_WEIGHT_DECAY * tf.reduce_sum(var ** 2) for var in train_actor_vars])

    optim_actor = tf.train.AdamOptimizer(ACTOR_LR)
    loss_actor = -tf.reduce_mean(train_critic_current_action) + weight_decay_actor

    # Actor Optimization
    grads_and_vars_actor = optim_actor.compute_gradients(loss_actor, var_list=train_actor_vars,
                                                         colocate_gradients_with_ops=True)
    optimize_actor = optim_actor.apply_gradients(grads_and_vars_actor)

    with tf.control_dependencies([optimize_actor]):
        train_target_vars = zip(train_actor_vars, target_actor_vars)
        train_actor = tf.group(
            *[target.assign(TAU * train + (1 - TAU) * target) for train, target in train_target_vars])

with tf.name_scope('critic_loss'):
    q_target_value = tf.select(done_placeholder, reward_placeholder, reward_placeholder + GAMMA * tf.stop_gradient(target_critic_next_output))
    q_error = tf.abs(q_target_value - train_critic_placeholder_action)
    q_error_batch = tf.reduce_mean(q_error)
    weight_decay_critic = tf.add_n([CRITIC_L2_WEIGHT_DECAY * tf.reduce_sum(var ** 2) for var in train_critic_vars])
    loss_critic = q_error_batch + weight_decay_critic

    # Critic Optimization
    optim_critic = tf.train.AdamOptimizer(CRITIC_LR)
    grads_and_vars_critic = optim_critic.compute_gradients(loss_critic, var_list=train_critic_vars, colocate_gradients_with_ops=True)
    optimize_critic = optim_critic.apply_gradients(grads_and_vars_critic)
    with tf.control_dependencies([optimize_critic]):
        train_target_vars = zip(train_critic_vars, target_critic_vars)
        train_critic = tf.group(
            *[target.assign(TAU * train + (1 - TAU) * target) for train, target in train_target_vars])

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                        allow_soft_placement=True,
                                        log_device_placement=False))
sess.run(tf.global_variables_initializer())

# Initialize target = train
sess.run([target.assign(train) for train, target in zip(train_actor_vars, target_actor_vars)])
sess.run([target.assign(train) for train, target in zip(train_critic_vars, target_critic_vars)])

sess.graph.finalize()

replay_buffer = Memory(1, REPLAY_BUFFER_SIZE, env)
rewards = []

for episode in tqdm(range(10000)):
    env_state = env.reset()
    eta_noise = Noise()
    training = replay_buffer.count >= min(ITERATIONS_BEFORE_TRAINING, REPLAY_BUFFER_SIZE)
    testing = (episode + 1) % 2 == 0 and training
    history = []
    total_reward = 0

    for step in tqdm(range(MAX_EPISODE_LENGTH)):
        action, q = sess.run([train_actor_output, train_critic_current_action], feed_dict={k: [[v]] for k, v in zip(state_placeholders, env_state)})

        action = action[0]

        action = action if testing else eta_noise.reflected_ou(action * np.array([1, 1, 0, 1]), theta=[.15, .15, .75, .15], sigma=[.10, .10, .10, .10], min=-1, max=1)

        assert action.shape == env.action_space.sample().shape, (action.shape, env.action_space.sample().shape)

        max_xvel = 20
        max_yvel = 8
        max_yawrate = 0.2
        max_altitude = 15
        action = np.clip(action, -1, 1) * np.array([max_xvel, max_yvel, max_yawrate, max_altitude / 4.0]) - np.array([0, 0, 0, max_altitude])

        env_next_state, env_reward, env_done, env_info = env.step(action)
        replay_buffer.add(env_state, env_reward, action, env_done, priority=300)

        env_state = env_next_state

        total_reward += env_reward

        if training:
            states_batch, action_batch, reward_batch, next_states_batch, done_batch, indexes = replay_buffer.sample(BATCH_SIZE, prioritized=True)

            feed = {
                action_placeholder: action_batch,
                reward_placeholder: reward_batch,
                done_placeholder: done_batch
            }

            feed.update({k: v for k, v in zip(state_placeholders, states_batch)})
            feed.update({k: v for k, v in zip(next_state_placeholders, next_states_batch)})

            _, _, errors, critic_error = sess.run([train_critic, train_actor, q_error, q_error_batch], feed_dict=feed)

            replay_buffer.update(indexes, errors)

            print 'q:{:5f} reward:{:5f} trainerror:{:5f}'.format(q[0], env_reward, critic_error)

        if env_done:
            break

    print 'Total Reward', total_reward