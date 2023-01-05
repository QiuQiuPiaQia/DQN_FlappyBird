import tensorflow as tf
import numpy as np
import cv2

ACTIONS = 2  # number of valid actions


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


def load_weights():
    # 加载模型参数
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    return sess


def do_nothing():
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    return do_nothing


class Agent(object):
    def __init__(self):
        self.s, self.readout, self.h_fc1 = createNetwork()
        self.sess = load_weights()
        self.do_nothing = do_nothing()
        self.s_t = None

    def make_decision(self, x_t):
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        # 使用神经网络进行inference，根据神经网络输出构建action
        readout_t = self.sess.run(self.readout, feed_dict={self.s: [self.image_to_frames(x_t)]})[0]

        action_index = np.argmax(readout_t)
        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1

        return readout_t, action_index, a_t

    def image_to_frames(self, x_t):
        if self.s_t is None:
            self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        else:
            x_t = np.reshape(x_t, (80, 80, 1))
            self.s_t = np.append(x_t, self.s_t[:, :, :3], axis=2)
        return self.s_t

