import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

class Network():
    def __init__(self, image_width=320, image_height=240, training_rate=0.000001,
                 channels=1, n_input_data=10, n_output_data=4, log_location="./.tflogs"):

        #      CONSTANTS     #
        #--------------------#
        self.IMG_HEIGHT = image_height
        self.IMG_WIDTH = image_width
        self.CHANNELS = channels
        self.TRAINING_RATE = training_rate
        self.N_INPUT_DATA = n_input_data
        self.N_OUTPUT_DATA = n_output_data
        self.LOG_LOCATION = log_location

        #--------------------#
        #   NETWORK LAYOUT   #
        #--------------------#
        self.conv_layout = [
            {"inputs": None, "num_outputs": 8, "kernel_size": 5, "stride": 1, "padding": "SAME", "normalizer_fn": None, "scope": "Conv_0", "biases_initializer": tf.constant_initializer(0.1)},
            {"inputs": None, "num_outputs": 16, "kernel_size": 4, "stride": 4, "padding": "VALID", "normalizer_fn": None, "scope": "Conv_1", "biases_initializer": tf.constant_initializer(0.1)},
            {"inputs": None, "num_outputs": 32, "kernel_size": 5, "stride": 1, "padding": "SAME", "normalizer_fn": None, "scope": "Conv_2", "biases_initializer": tf.constant_initializer(0.1)},
            {"inputs": None, "num_outputs": 64, "kernel_size": 4, "stride": 4, "padding": "VALID", "normalizer_fn": None, "scope": "Conv_3", "biases_initializer": tf.constant_initializer(0.1)}
        ]

        self.img_fc_layout = [
            {"inputs": None, "num_outputs": 500, "normalizer_fn": None, "scope": "ImgFC_0", "biases_initializer": tf.constant_initializer(0.1)}
        ]

        self.data_fc_layout = [
            {"inputs": None, "num_outputs": 500, "normalizer_fn": None, "scope": "DataFC_0", "biases_initializer": tf.constant_initializer(0.1)},
            {"inputs": None, "num_outputs": 500, "normalizer_fn": None, "scope": "DataFC_1", "biases_initializer": tf.constant_initializer(0.1)}
        ]

        self.combined_fc_layout = [
            {"inputs": None, "num_outputs": 500, "normalizer_fn": None, "scope": "CombFC_0", "biases_initializer": tf.constant_initializer(0.1)},
            {"inputs": None, "num_outputs": 500, "normalizer_fn": None, "scope": "CombFC_1", "biases_initializer": tf.constant_initializer(0.1)},
            {"inputs": None, "num_outputs": 4, "activation_fn": None, "scope": "CombFC_2", "biases_initializer": tf.constant_initializer(0.1)}
        ]

        #--------------------#
        #    PLACEHOLDERS    #
        #--------------------#
        self.input_image = tf.placeholder(tf.float32, [None, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNELS])
        self.input_data  = tf.placeholder(tf.float32, [None, self.N_INPUT_DATA])
        self.desired_action = tf.placeholder(tf.float32, [None, self.N_OUTPUT_DATA])

        #--------------------#
        #  NETWORK CREATION  #
        #--------------------#
        # First half
        self.conv_list = self._create_conv(self.input_image)
        self.conv_out = self.conv_list[-1]
        self.data_list = self._create_data_fc(self.input_data)
        self.data_out = self.data_list[-1]
        self.combined = self.data_out + self.conv_out

        # Second half
        self.output_list = self._create_fcs(self.combined)
        self.output = self.output_list[-1]
        self.scale = tf.constant(np.array([1, 10, 10, 1]), tf.float32)
        self.scaled_output = self.output * self.scale

        #--------------------#
        # LOSS AND OPTIMIZER #
        #--------------------#
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs((self.desired_action - self.output)), reduction_indices=[1]))
        self.weighted_loss = tf.reduce_sum((self.desired_action * self.scale - self.scaled_output)**2)
        self.optimizer = tf.train.AdamOptimizer(self.TRAINING_RATE).minimize(self.loss)
        self.weighted_optimizer = tf.train.AdamOptimizer(self.TRAINING_RATE).minimize(self.weighted_loss)

        #--------------------#
        #  IMPORTANT THINGS  #
        #--------------------#
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.writer = tf.train.SummaryWriter(self.LOG_LOCATION, self.sess.graph)


    # NETWORK CREATING FUNCTIONS
    #=======================================================================================================================#

    def _create_conv(self, input):
        outs = [input]

        for x in self.conv_layout:
            x["inputs"] = outs[-1]
            outs.append(slim.conv2d(**x))
        
        outs.append(tf.contrib.layers.flatten(outs[-1]))
        
        for x in self.img_fc_layout:
            x["inputs"] = outs[-1]
            outs.append(slim.fully_connected(**x))

        return outs


    def _create_data_fc(self, input):
        outs = [input]

        for x in self.data_fc_layout:
            x["inputs"] = outs[-1]
            outs.append(slim.fully_connected(**x))
        
        return outs


    def _create_fcs(self, input):
        outs = [input]

        for x in self.combined_fc_layout:
            x["inputs"] = outs[-1]
            outs.append(slim.fully_connected(**x))
        
        return outs

    # GETTERS
    #=======================================================================================================================#
    def get_input(self, img, data):
        return self.sess.run((self.input_image, self.input_data), feed_dict={self.input_image: img, self.input_data: data})

    def get_output(self, img, data):
        return self.sess.run(self.output, feed_dict={self.input_image: img, self.input_data: data})

    # TRAINING
    #=======================================================================================================================#
    def train(self, img, data, desired, use_weighted_loss=False):
        if not use_weighted_loss:
            loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict={self.input_image: img, self.input_data: data, self.desired_action: desired})
        else:
            loss, _ = self.sess.run((self.weighted_loss, self.weighted_optimizer), feed_dict={self.input_image: img, self.input_data: data, self.desired_action: desired})
        return loss

    # USEFUL FUNCTIONS
    #=======================================================================================================================#
    def load_network(self, location):
        self.saver.restore(self.sess, location)

    def save_network(self, location):
        self.saver.save(self.sess, location)
