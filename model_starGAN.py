import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib


class ModelStarGAN(object):
    def __init__(self, IMG_CHANNEL, NUMBER_OF_DOMAIN, BASE_CHANNEL, seed, IMG_WIDTH, IMG_HEIGHT):
        self.IMG_CHANNEL = IMG_CHANNEL
        self.NUMBER_OF_DOMAIN = NUMBER_OF_DOMAIN
        self.BASE_CHANNEL = BASE_CHANNEL
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.SEED = seed
        np.random.seed(seed=self.SEED)


    def generator(self, x, nc, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.name_scope("concat"):
                concat = tf.concat([x, nc], axis=3)

            with tf.name_scope("G_layer1"): #layer1 conv (h, w, 3+nc) -> (h, w, 64)
                wg1 = tf.get_variable('wg1', [7, 7, self.IMG_CHANNEL + self.NUMBER_OF_DOMAIN, self.BASE_CHANNEL],
                                      initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32) #[h, w, 3+nc] -> [h, w, 64]
                bg1 = tf.get_variable('bg1', [self.BASE_CHANNEL], initializer=tf.constant_initializer(0.0))

                conv1 = tf.nn.conv2d(concat, wg1, strides=[1, 1, 1, 1], padding="SAME", name='G_conv1') + bg1
                #instance normalization
                in1 = contrib.layers.instance_norm(conv1)
                #relu
                rl1 = tf.nn.relu(in1)

            with tf.name_scope("G_layer2"): #layer2 conv (h, w, 64) -> (h/2, w/2, 128)
                wg2 = tf.get_variable('wg2', [4, 4, self.BASE_CHANNEL, self.BASE_CHANNEL * 2],
                                      initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32) #[h, w, 64 -> [h/2, w/2, 128]
                bg2 = tf.get_variable('bg2', [self.BASE_CHANNEL * 2], initializer=tf.constant_initializer(0.0))

                conv2 = tf.nn.conv2d(rl1, wg2, strides=[1, 2, 2, 1], padding="SAME", name='G_conv2') + bg2
                #instance normalization
                in2 = contrib.layers.instance_norm(conv2)
                #relu
                rl2 = tf.nn.relu(in2)

            with tf.name_scope("G_layer3"): #layer3 conv (h/2, w/2, 128) -> (h/4, w/4, 256)
                wg3 = tf.get_variable('wg3', [4, 4, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4],
                                      initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32) #[h, w, 64 -> [h/2, w/2, 128]
                bg3 = tf.get_variable('bg3', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))

                conv3 = tf.nn.conv2d(rl2, wg3, strides=[1, 2, 2, 1], padding="SAME", name='G_conv3') + bg3
                #instance normalization
                in3 = contrib.layers.instance_norm(conv3)
                #relu
                rl3 = tf.nn.relu(in3)

            with tf.name_scope("G_layer4"): # layer4 residual block (h/4, w/4, 256) -> (h/4, w/4, 256)
                wg4 = tf.get_variable('wg4', [3, 3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg4 = tf.get_variable('bg4', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))
                conv4 = tf.nn.conv2d(rl3, wg4, strides=[1, 1, 1, 1], padding="SAME", name='G_conv4') + bg4
                # instance normalization
                in4 = contrib.layers.instance_norm(conv4)
                # add
                add4 = tf.add(rl3, in4)
                # relu
                rl4 = tf.nn.relu(add4)

            with tf.name_scope("G_layer5"): # layer5 residual block (h/4, w/4, 256) -> (h/4, w/4, 256)
                wg5 = tf.get_variable('wg5', [3, 3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg5 = tf.get_variable('bg5', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))
                conv5 = tf.nn.conv2d(rl4, wg5, strides=[1, 1, 1, 1], padding="SAME", name='G_conv5') + bg5
                # instance normalization
                in5 = contrib.layers.instance_norm(conv5)
                # add
                add5 = tf.add(rl4, in5)
                # relu
                rl5 = tf.nn.relu(add5)

            with tf.name_scope("G_layer6"): # layer6 residual block (h/4, w/4, 256) -> (h/4, w/4, 256)
                wg6 = tf.get_variable('wg6', [3, 3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg6 = tf.get_variable('bg6', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))
                conv6 = tf.nn.conv2d(rl5, wg6, strides=[1, 1, 1, 1], padding="SAME", name='G_conv6') + bg6
                # instance normalization
                in6 = contrib.layers.instance_norm(conv6)
                # add
                add6 = tf.add(rl5, in6)
                # relu
                rl6 = tf.nn.relu(add6)

            with tf.name_scope("G_layer7"): # layer7 residual block (h/4, w/4, 256) -> (h/4, w/4, 256)
                wg7 = tf.get_variable('wg7', [3, 3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg7 = tf.get_variable('bg7', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))
                conv7 = tf.nn.conv2d(rl6, wg7, strides=[1, 1, 1, 1], padding="SAME", name='G_conv7') + bg7
                # instance normalization
                in7 = contrib.layers.instance_norm(conv7)
                # add
                add7 = tf.add(rl6, in7)
                # relu
                rl7 = tf.nn.relu(add7)

            with tf.name_scope("G_layer8"): # layer8 residual block (h/4, w/4, 256) -> (h/4, w/4, 256)
                wg8 = tf.get_variable('wg8', [3, 3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg8 = tf.get_variable('bg8', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))
                conv8 = tf.nn.conv2d(rl7, wg8, strides=[1, 1, 1, 1], padding="SAME", name='G_conv8') + bg8
                # instance normalization
                in8 = contrib.layers.instance_norm(conv8)
                # add
                add8 = tf.add(rl7, in8)
                # relu
                rl8 = tf.nn.relu(add8)

            with tf.name_scope("G_layer9"): # layer9 residual block (h/4, w/4, 256) -> (h/4, w/4, 256)
                wg9 = tf.get_variable('wg9', [3, 3, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg9 = tf.get_variable('bg9', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))
                conv9 = tf.nn.conv2d(rl8, wg9, strides=[1, 1, 1, 1], padding="SAME", name='G_conv9') + bg9
                # instance normalization
                in9 = contrib.layers.instance_norm(conv9)
                # add
                add9 = tf.add(rl8, in9)
                # relu
                rl9 = tf.nn.relu(add9)

            with tf.name_scope("G_layer10"): # layer10 deconv (h/4, w/4, 256) -> (h/2, w/2, 128)
                wg10 = tf.get_variable('wg10', [4, 4, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg10 = tf.get_variable('bg10', [self.BASE_CHANNEL * 2], initializer=tf.constant_initializer(0.0))

                output_shape10 = tf.stack(
                    # [tf.shape(lR3)[0], tf.shape(lR3)[1], tf.shape(lR3)[2], tf.shape(lR3)[3]])
                [tf.shape(rl9)[0], tf.shape(rl9)[1] * 2, tf.shape(rl9)[2] * 2, tf.div(tf.shape(rl9)[3], tf.constant(2))])
                deconv10 = tf.nn.conv2d_transpose(rl9, wg10, output_shape=output_shape10, strides=[1, 2, 2, 1],
                                                 padding="SAME") + bg10
                # instance normalization
                in10 = contrib.layers.instance_norm(deconv10)
                # relu
                rl10 = tf.nn.relu(in10)

            with tf.name_scope("G_layer11"):  # layer11 deconv (h/2, w/2, 128) -> (h, w, 64)
                wg11 = tf.get_variable('wg11', [4, 4, self.BASE_CHANNEL, self.BASE_CHANNEL * 2],
                                       initializer=tf.random_normal_initializer
                                       (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bg11 = tf.get_variable('bg11', [self.BASE_CHANNEL], initializer=tf.constant_initializer(0.0))

                output_shape11 = tf.stack(
                    # [tf.shape(lR3)[0], tf.shape(lR3)[1], tf.shape(lR3)[2], tf.shape(lR3)[3]])
                    [tf.shape(rl10)[0], tf.shape(rl10)[1] * 2, tf.shape(rl10)[2] * 2, tf.div(tf.shape(rl10)[3], tf.constant(2))])
                deconv11 = tf.nn.conv2d_transpose(rl10, wg11, output_shape=output_shape11, strides=[1, 2, 2, 1],
                                                  padding="SAME") + bg11
                # instance normalization
                in11 = contrib.layers.instance_norm(deconv11)
                # relu
                rl11 = tf.nn.relu(in11)

            with tf.name_scope("G_layer12"):  # layer12 conv (h, w, 64) -> (h, w, 3)
                wg12 = tf.get_variable('wg12', [7, 7, self.BASE_CHANNEL, self.IMG_CHANNEL], initializer=tf.random_normal_initializer
                                      (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)  # [h, w, 3+nc] -> [h, w, 64]
                bg12 = tf.get_variable('bg12', [self.IMG_CHANNEL], initializer=tf.constant_initializer(0.0))

                conv12 = tf.nn.conv2d(rl11, wg12, strides=[1, 1, 1, 1], padding="SAME", name='G_conv12') + bg12
                # tanh
                tanh12 = tf.nn.tanh(conv12)

            return tanh12


    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):

            with tf.name_scope("D_layer1"): # layer1 conv1
                wd1 = tf.get_variable('wd1', [4, 4, self.IMG_CHANNEL, self.BASE_CHANNEL], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd1 = tf.get_variable('bd1', [self.BASE_CHANNEL], initializer=tf.constant_initializer(0.0))

                conv1 = tf.nn.conv2d(x, wd1, strides=[1, 2, 2, 1], padding="SAME", name='D_conv1') + bd1
                # leakyReLU function
                lr1 =self.leaky_relu(conv1, alpha=0.01)

            with tf.name_scope("D_layer2"): # layer2 conv2
                wd2 = tf.get_variable('wd2', [4, 4, self.BASE_CHANNEL, self.BASE_CHANNEL * 2], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd2 = tf.get_variable('bd2', [self.BASE_CHANNEL * 2], initializer=tf.constant_initializer(0.0))

                conv2 = tf.nn.conv2d(lr1, wd2, strides=[1, 2, 2, 1], padding="SAME", name='D_conv2') + bd2
                # leakyReLU function
                lr2 =self.leaky_relu(conv2, alpha=0.01)

            with tf.name_scope("D_layer3"): # layer3 conv3
                wd3 = tf.get_variable('wd3', [4, 4, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd3 = tf.get_variable('bd3', [self.BASE_CHANNEL * 4], initializer=tf.constant_initializer(0.0))

                conv3 = tf.nn.conv2d(lr2, wd3, strides=[1, 2, 2, 1], padding="SAME", name='D_conv3') + bd3
                # leakyReLU function
                lr3 =self.leaky_relu(conv3, alpha=0.01)

            with tf.name_scope("D_layer4"): # layer4 conv4
                wd4 = tf.get_variable('wd4', [4, 4, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd4 = tf.get_variable('bd4', [self.BASE_CHANNEL * 8], initializer=tf.constant_initializer(0.0))

                conv4 = tf.nn.conv2d(lr3, wd4, strides=[1, 2, 2, 1], padding="SAME", name='D_conv4') + bd4
                # leakyReLU function
                lr4 =self.leaky_relu(conv4, alpha=0.01)

            with tf.name_scope("D_layer5"): # layer5 conv5
                wd5 = tf.get_variable('wd5', [4, 4, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 16], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd5 = tf.get_variable('bd5', [self.BASE_CHANNEL * 16], initializer=tf.constant_initializer(0.0))

                conv5 = tf.nn.conv2d(lr4, wd5, strides=[1, 2, 2, 1], padding="SAME", name='D_conv5') + bd5
                # leakyReLU function
                lr5 =self.leaky_relu(conv5, alpha=0.01)

            with tf.name_scope("D_layer6"): # layer6 conv6
                wd6 = tf.get_variable('wd6', [4, 4, self.BASE_CHANNEL * 16, self.BASE_CHANNEL * 32], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd6 = tf.get_variable('bd6', [self.BASE_CHANNEL * 32], initializer=tf.constant_initializer(0.0))

                conv6 = tf.nn.conv2d(lr5, wd6, strides=[1, 2, 2, 1], padding="SAME", name='D_conv6') + bd6
                # leakyReLU function
                lr6 =self.leaky_relu(conv6, alpha=0.01)

            with tf.name_scope("D_layer7"): # layer7 conv7 output about real/fake
                wd7 = tf.get_variable('wd7', [4, 4, self.BASE_CHANNEL * 32, 1], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd7 = tf.get_variable('bd7', [1], initializer=tf.constant_initializer(0.0))

                out7 = tf.nn.conv2d(lr6, wd7, strides=[1, 2, 2, 1], padding="SAME", name='D_conv7') + bd7
                # out7 = tf.reduce_mean(conv7, axis=(1, 2), name='D_rm7')

            with tf.name_scope("D_layer8"): # layer8 conv8 output about domain
                wd8 = tf.get_variable('wd8', [self.IMG_WIDTH // self.BASE_CHANNEL, self.IMG_HEIGHT // self.BASE_CHANNEL, self.BASE_CHANNEL * 32,
                                              self.NUMBER_OF_DOMAIN], initializer=tf.random_normal_initializer
                (mean=0.0, stddev=0.02, seed=self.SEED), dtype=tf.float32)
                bd8 = tf.get_variable('bd8', [self.NUMBER_OF_DOMAIN], initializer=tf.constant_initializer(0.0))

                conv8 = tf.nn.conv2d(lr6, wd8, strides=[1, 1, 1, 1], padding="SAME", name='D_conv8') + bd8

                red8 = tf.reduce_mean(conv8, axis=(1, 2), name='D_rm8')
                out8 = tf.nn.softmax(red8)
            return out7, out8

    def leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


