import tensorflow as tf


def DPM(x, in_channel, g, name):
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(x, in_channel, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(conv1, in_channel+g, 3, 1, 'same', activation=None, name='conv2')
        res, dense = tf.split(conv2,[in_channel,g],-1)
        out = tf.concat([x[:, :, :, :in_channel] + res, x[:, :, :, in_channel:], dense],axis=-1, name='concat')
        return out

def SE(x, channel_in, channel_h, name):
    with tf.variable_scope(name):
        avg = tf.reduce_mean(x, axis=[1, 2])
        a_1 = tf.layers.dense(avg, channel_h, name='w0', activation=tf.nn.relu)
        a_2 = tf.layers.dense(a_1, channel_in, name='w1')
        att = tf.nn.sigmoid(a_2)
    return att

def DPB(x, in_channel, g, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        dpm1 = DPM(x,in_channel, g, 'DPM1')
        dpm2 = DPM(dpm1,in_channel, g,'DPM2')
        dpm3 = DPM(dpm2,in_channel, g,'DPM3')
        dpm4 = DPM(dpm3,in_channel, g,'DPM4')
        bottleneck = tf.layers.conv2d(dpm4, in_channel + g, 1, 1, 'same', name='bottleneck')
        att_c = tf.expand_dims(tf.expand_dims(SE(bottleneck, in_channel + g, 32, 'att_c'),1),1)
        out = tf.multiply(bottleneck,att_c)
    return out

def DPDN(x, channel, g, D):
    with tf.variable_scope('DPDN'):
        x_ = tf.space_to_depth(x,2,'PS')
        conv1 = tf.layers.conv2d(x_, channel, 3, 1, 'same', name='conv1')
        conv2 = tf.layers.conv2d(conv1, channel, 3, 1, 'same', name='conv2')
        dpb = DPB(conv2, channel, g, 'DPB0')
        for i in range(1,D):
            dpb_ = DPB(dpb, channel, g, 'DPB' + str(i))
            res, dense = tf.split(dpb_, [channel, g], -1)
            dpb = tf.concat([dpb[:, :, :, :channel] + res, dpb[:, :, :, channel:], dense],axis=-1, name='concat'+ str(i))
        ds = tf.layers.conv2d(dpb, channel, 1, 1, 'same', name='DS')
        df = tf.layers.conv2d(ds, channel, 3, 1, 'same', name='GF') + conv1
        out = tf.layers.conv2d(df, 12, 3, 1, 'same', name='conv_last') + x_
        out = tf.depth_to_space(out,2)
    return out
