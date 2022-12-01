import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

###### 64x64x3
def generator(inputs, is_train=True, reuse=False):
    FLAGS = tf.app.flags.FLAGS
    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    gf_dim = 128 # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    assert FLAGS.image_size == image_size#, print("image size should be 64")

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init, b_init=b_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)

    return net_h4, logits


def discriminator(inputs, is_train=True, reuse=False):
    FLAGS = tf.app.flags.FLAGS
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/output_real_fake')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)

        net_h5 = FlattenLayer(net_h3, name='d/h5/flatten')
        net_h5 = DenseLayer(net_h5, n_units=2, act=tf.identity,
                W_init = w_init, name='d/h5/output_classes')
        logits2 = net_h5.outputs
        net_h5.outputs = tf.nn.softmax(net_h5.outputs)
    return net_h4, logits, net_h5, logits2, net_h3


def imageEncoder(inputs, is_train=True, reuse=False):
    # Same architecure as the discriminator, different last layer
    FLAGS = tf.app.flags.FLAGS
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("imageEncoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='p/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=b_init, name='p/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='p/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=FLAGS.z_dim,
                act=tf.identity,
                # act=tf.nn.tanh,
                W_init = w_init, name='p/h4/output_real_fake')

    return net_h4


# def imageEncoder_old(inputs, output_dim = 100, is_train=True, reuse=False):
#     """ CNN part of VGG19, modified from tensorlayer/example/tutorial_vgg19.py
#     """
#     w_init = tf.random_normal_initializer(stddev=0.02)
#
#     with tf.variable_scope("imageEncoder", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         network = tl.layers.InputLayer(inputs, name='imageEncoder/input_layer')
#
#         network = Conv2d(network, 64, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h1_1/conv2d')
#         network = Conv2d(network, 64, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h1_2/conv2d')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
#                     padding='SAME', name='imageEncoder/h1/MaxPool2d')
#
#         network = Conv2d(network, 128, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h2_1/conv2d')
#         network = Conv2d(network, 128, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h2_2/conv2d')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
#                     padding='SAME', name='imageEncoder/h2/MaxPool2d')
#
#         network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h3_1/conv2d')
#         network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h3_2/conv2d')
#         network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h3_3/conv2d')
#         network = Conv2d(network, 256, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h3_4/conv2d')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
#                     padding='SAME', name='imageEncoder/h3/MaxPool2d')
#
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h4_1/conv2d')
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h4_2/conv2d')
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h4_3/conv2d')
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h4_4/conv2d')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
#                     padding='SAME', name='imageEncoder/h4/MaxPool2d')
#
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h5_1/conv2d')
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h5_2/conv2d')
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h5_3/conv2d')
#         network = Conv2d(network, 512, (3, 3), (1, 1), act=tf.nn.relu,
#                     padding='SAME', W_init=w_init, name='imageEncoder/h5_4/conv2d')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
#                     padding='SAME', name='imageEncoder/h5/MaxPool2d')
#
#         network = FlattenLayer(network, name='imageEncoder/flatten')
#
#         network = DenseLayer(network, n_units=output_dim, act=tf.identity,
#                 W_init = w_init, name='imageEncoder/reduced_output')
#
#     return network


################## 256x256x3

def generator_256(inputs, is_train=True, reuse=False):
    FLAGS = tf.app.flags.FLAGS
    image_size = 256
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    s32, s64 = int(image_size/32), int(image_size/64)   # for 256
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    assert FLAGS.image_size == image_size#, print("image size should be 256")

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s64*s64, W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s64, s64, gf_dim*8], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*16, (5, 5), out_size=(s32, s32), strides=(2, 2),           # add for 64-->256
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*8, (5, 5), out_size=(s16, s16), strides=(2, 2),            # add for 64-->256
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        net_h4 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h4/batch_norm')

        net_h5 = DeConv2d(net_h4, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h5/decon2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h5/batch_norm')

        net_h6 = DeConv2d(net_h5, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h6/decon2d')
        logits = net_h6.outputs
        net_h6.outputs = tf.nn.tanh(net_h6.outputs)

    return net_h6, logits


def discriminator_256(inputs, is_train=True, reuse=False):
    FLAGS = tf.app.flags.FLAGS
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h3 = Conv2d(net_h3, df_dim*16, (5, 5), (2, 2), act=None,                # add for 64-->256
                padding='SAME', W_init=w_init, name='d/h3_2/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3_2/batch_norm')

        net_h3 = Conv2d(net_h3, df_dim*32, (5, 5), (2, 2), act=None,                # add for 64-->256
                padding='SAME', W_init=w_init, name='d/h3_3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3_3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/output_real_fake')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)

        net_h5 = FlattenLayer(net_h3, name='d/h5/flatten')
        net_h5 = DenseLayer(net_h5, n_units=2, act=tf.identity,
                W_init = w_init, name='d/h5/output_classes')
        logits2 = net_h5.outputs
        net_h5.outputs = tf.nn.softmax(net_h5.outputs)
    return net_h4, logits, net_h5, logits2, net_h3


def imageEncoder_256(inputs, is_train=True, reuse=False):
    # Same architecure as the discriminator, different last layer
    FLAGS = tf.app.flags.FLAGS
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("imageEncoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='p/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='p/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h3/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*16, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),  # add for 64-->256
                padding='SAME', W_init=w_init, name='p/h3_2/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h3_2/batch_norm')

        net_h3 = Conv2d(net_h3, df_dim*16, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),  # add for 64-->256
                padding='SAME', W_init=w_init, name='p/h3_3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='p/h3_3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='p/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=FLAGS.z_dim,
                act=tf.identity,    # for gaussian distribution
                # act=tf.nn.tanh,   # for uniform distribution
                W_init = w_init, name='p/h4/output_real_fake')

    return net_h4
