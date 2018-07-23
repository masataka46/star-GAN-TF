import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib as contrib
from PIL import Image
import utility as Utility
import argparse

from make_datasets_many import Make_datasets_many
from model_starGAN import ModelStarGAN

def parser():
    parser = argparse.ArgumentParser(description='train LSGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log180720  ', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=1000, help='epoch')
    parser.add_argument('--dir_name1', '-dn1', type=str, default='/PATH/TO/DOMAIN1_DIRECTORY/',
                        help='directory name of domain1')
    parser.add_argument('--dir_name2', '-dn2', type=str, default='/PATH/TO/DOMAIN2_DIRECTORY/',
                        help='directory name of domain2')
    parser.add_argument('--dir_name3', '-dn3', type=str, default='/PATH/TO/DOMAIN3_DIRECTORY/',
                        help='directory name of domain3')

    return parser.parse_args()

args = parser()


#global variants
BATCH_SIZE = args.batch_size
LOGFILE_NAME = args.log_file_name
EPOCH = args.epoch
DIR_NAME1 = args.dir_name1
DIR_NAME2 = args.dir_name2
DIR_NAME3 = args.dir_name3

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3
BASE_CHANNEL = 64
NOISE_UNIT_NUM = 100
NOISE_MEAN = 0.0
NOISE_STDDEV = 1.0
TEST_DATA_SAMPLE = 4 * 3
L2_NORM = 0.001
KEEP_PROB_RATE = 0.5
LAMBDA_CLS = 1.0
LAMBDA_REC = 10.0
SEED = 1234
np.random.seed(seed=SEED)
NUMBER_OF_DOMAIN = 3 #=class number
BOARD_DIR_NAME = './tensorboard/' + LOGFILE_NAME

out_image_dir = './out_images_starGAN' #output image file
out_model_dir = './out_models_starGAN' #output model file

try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    pass

make_datasets = Make_datasets_many(DIR_NAME1, DIR_NAME2, DIR_NAME3, IMG_WIDTH, IMG_HEIGHT, SEED)

# def leaky_relu(x, alpha):
#     return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# def gaussian_noise(input, std): #used at discriminator
#     noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=SEED)
#     return input + noise

model = ModelStarGAN(IMG_CHANNEL, NUMBER_OF_DOMAIN, BASE_CHANNEL, SEED, IMG_WIDTH, IMG_HEIGHT)

# placeholder
x_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='x_') #images of domain1 & domain2
x_label_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, NUMBER_OF_DOMAIN], name='x_label_') #labels of changed domain
x_label_rev_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, NUMBER_OF_DOMAIN], name='x_label_rev') #labels of original domain
x_real_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='x_real_') #real images
tar_dis_g_adv_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_g_adv_') #real/fake-target of discriminator related to generator
tar_dis_r_adv_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_r_adv_') #real/fake-target of discriminator related to real image
tar_dis_g_cls_ = tf.placeholder(tf.float32, [None, NUMBER_OF_DOMAIN], name='d_dis_g_cls_') #domain-target of discriminator related to generator
tar_dis_r_cls_ = tf.placeholder(tf.float32, [None, NUMBER_OF_DOMAIN], name='d_dis_r_cls_') #domain-target of discriminator related to real image

# stream around generator
x_gen = model.generator(x_, x_label_, reuse=False) #
x_gen_gen = model.generator(x_gen, x_label_rev_, reuse=True)

#stream around discriminator
out_dis_g_adv, out_dis_g_cls = model.discriminator(x_gen, reuse=False) #from generator
out_dis_r_adv, out_dis_r_cls = model.discriminator(x_, reuse=True) #real image

with tf.name_scope("loss"):
    #adversarial loss
    tar_dis_g_adv_re = tf.reshape(tar_dis_g_adv_, [tf.shape(tar_dis_g_adv_)[0], 1, 1, 1])
    shape_g = tf.stack([tf.constant(1), tf.shape(out_dis_g_adv)[1], tf.shape(out_dis_g_adv)[1], tf.constant(1)])
    tar_dis_g_adv_tile = tf.tile(tar_dis_g_adv_re, shape_g)
    tar_dis_r_adv_re = tf.reshape(tar_dis_r_adv_, [tf.shape(tar_dis_r_adv_)[0], 1, 1, 1])
    shape_r = tf.stack([tf.constant(1), tf.shape(out_dis_r_adv)[1], tf.shape(out_dis_r_adv)[1], tf.constant(1)])
    tar_dis_r_adv_tile = tf.tile(tar_dis_r_adv_re, shape_r)
    # loss_dis_g_adv = tf.reduce_mean(tf.square(out_dis_g_adv - tar_dis_g_adv_), name='Loss_dis_gen_adv') #loss related to generator, adv
    # loss_dis_r_adv = tf.reduce_mean(tf.square(out_dis_r_adv - tar_dis_r_adv_), name='Loss_dis_rea_adv') #loss related to real imaeg, adv
    # TODO change to Wasserstein Loss...now, least square loss
    loss_dis_g_adv = tf.reduce_mean(tf.square(out_dis_g_adv - tar_dis_g_adv_tile), name='Loss_dis_gen_adv') #loss related to generator, adv
    loss_dis_r_adv = tf.reduce_mean(tf.square(out_dis_r_adv - tar_dis_r_adv_tile), name='Loss_dis_rea_adv') #loss related to real imaeg, adv

    #domain classification loss
    loss_dis_g_cls = - tf.reduce_mean(tar_dis_g_cls_ * tf.log(tf.clip_by_value(out_dis_g_cls, 1e-10, 1e+30)), name='Loss_dis_gen_cls') #cls loss in case generated image
    loss_dis_r_cls = - tf.reduce_mean(tar_dis_r_cls_ * tf.log(tf.clip_by_value(out_dis_r_cls, 1e-10, 1e+30)), name='Loss_dis_rea_cls') #cls loss in case real image

    #reconstruction loss
    loss_rec = tf.reduce_mean(tf.abs(x_ - x_gen_gen))

    #total loss of discriminator
    loss_dis_total =  loss_dis_g_adv + loss_dis_r_adv + LAMBDA_CLS * loss_dis_r_cls
    #total loss of generator
    loss_gen_total = loss_dis_g_adv +  LAMBDA_CLS * loss_dis_g_cls + LAMBDA_REC * loss_rec


tf.summary.scalar('loss_dis_total', loss_dis_total)
tf.summary.scalar('loss_gen_total', loss_gen_total)
merged = tf.summary.merge_all()


with tf.name_scope("train"):
    # t_vars = tf.trainable_variables()
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
    train_dis = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_dis_total, var_list=d_vars
                                        # var_list=[wd1, wd2, wd3, wd4, wd5, wd6, bd1, bd2, bd3, bd4, bd5, bd6]
                                                                                , name='Adam_dis')
    train_gen = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_gen_total, var_list=g_vars
                                        # var_list=[wg1, wg3, wg5, bg1, bg3, bg5, betag2, scaleg2, betag4, scaleg4]
                                                                                , name='Adam_gen')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(BOARD_DIR_NAME, sess.graph)

#training loop
for epoch in range(0, EPOCH):
    sum_loss_gen = np.float32(0)
    sum_loss_dis = np.float32(0)
    sum_loss_dis_r_cls = np.float32(0)
    sum_loss_dis_g_cls = np.float32(0)
    sum_loss_rec = np.float32(0)
    sum_loss_dis_r_adv = np.float32(0)
    sum_loss_dis_g_adv = np.float32(0)
    sum_loss_dis_g_adv_r = np.float32(0)
    sum_loss_dis_g_adv_f = np.float32(0)

    len_data = make_datasets.make_data_for_1_epoch()

    for i in range(0, len_data, BATCH_SIZE):
        # print("i, ", i)
        img_batch, class_chan_f, class_chan_r = make_datasets.get_data_for_1_batch(i, BATCH_SIZE)
        # z = make_datasets.make_random_z_with_norm(NOISE_MEAN, NOISE_STDDEV, len(img_batch), NOISE_UNIT_NUM)
        tar_dis_1 = make_datasets.make_target_1_0(1.0, len(img_batch))
        tar_dis_0 = make_datasets.make_target_1_0(0.0, len(img_batch))

        tar_dis_f_cls, tar_dis_r_cls = make_datasets.make_target_for_DC_loss_from_channel(class_chan_f, class_chan_r)

        #train discriminator
        sess.run(train_dis, feed_dict={x_:img_batch, x_label_:class_chan_f, x_real_:img_batch,
                    tar_dis_g_adv_:tar_dis_0, tar_dis_r_adv_:tar_dis_1, tar_dis_r_cls_:tar_dis_r_cls})

        #train generator
        sess.run(train_gen, feed_dict={x_:img_batch, x_label_:class_chan_f, x_label_rev_:class_chan_r,
                    tar_dis_g_adv_:tar_dis_1, tar_dis_g_cls_:tar_dis_f_cls})

        loss_gen_total_, loss_dis_g_adv_f_, loss_dis_g_cls_, loss_rec_ = sess.run([loss_gen_total, loss_dis_g_adv,
            loss_dis_g_cls, loss_rec], feed_dict={x_:img_batch,
            x_label_:class_chan_f, x_label_rev_:class_chan_r, tar_dis_g_adv_:tar_dis_1, tar_dis_g_cls_:tar_dis_f_cls})

        loss_dis_total_, loss_dis_r_adv_, loss_dis_r_cls_, loss_dis_g_adv_r_ = sess.run([loss_dis_total, loss_dis_r_adv,
                    loss_dis_r_cls, loss_dis_g_adv], feed_dict={x_:img_batch, x_label_:class_chan_f, x_real_:img_batch,
                    tar_dis_g_adv_:tar_dis_0, tar_dis_r_adv_:tar_dis_1, tar_dis_r_cls_:tar_dis_r_cls})

        #for tensorboard
        merged_ = sess.run(merged, feed_dict={x_:img_batch, x_label_:class_chan_f, x_real_:img_batch, x_label_rev_:class_chan_r,
                    tar_dis_g_adv_:tar_dis_0, tar_dis_r_adv_:tar_dis_1, tar_dis_r_cls_:tar_dis_r_cls, tar_dis_g_cls_:tar_dis_f_cls})

        summary_writer.add_summary(merged_, epoch)

        sum_loss_gen += loss_gen_total_
        sum_loss_dis += loss_dis_total_
        sum_loss_dis_r_cls += loss_dis_r_cls_
        sum_loss_dis_g_cls += loss_dis_g_cls_
        sum_loss_rec += loss_rec_
        sum_loss_dis_r_adv += loss_dis_r_adv_
        sum_loss_dis_g_adv_r += loss_dis_g_adv_r_
        sum_loss_dis_g_adv_f += loss_dis_g_adv_f_

    print("----------------------------------------------------------------------")
    print("epoch = {:}, Generator Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}".format(
        epoch, sum_loss_gen / len_data, sum_loss_dis / len_data))
    print("Generator : classifier loss = {:.4f}, adversarial loss = {:.4f}, reconstruction loss = {:.4f}".format(
        sum_loss_dis_g_cls / len_data, sum_loss_dis_g_adv_f / len_data, sum_loss_rec / len_data))
    print("Discriminator : classifier loss = {:.4f}, adversarial(real) loss = {:.4f}, adversarial(gen) loss = {:.4f}".format(
        sum_loss_dis_r_cls / len_data, sum_loss_dis_r_adv / len_data, sum_loss_dis_g_adv_r / len_data))


    if epoch % 10 == 0:
        img_batch, class_chan_f, class_chan_r = make_datasets.get_test_data_for_1_batch(10, int(TEST_DATA_SAMPLE // 3))
        gen_images, gen_gen_images = sess.run([x_gen, x_gen_gen], feed_dict={x_:img_batch, x_label_:class_chan_f, x_label_rev_:class_chan_r})
        Utility.make_output_img(img_batch, gen_images, gen_gen_images, int(TEST_DATA_SAMPLE // 3) ,out_image_dir, epoch, LOGFILE_NAME)


