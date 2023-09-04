import os
import numpy as np
import tensorflow as tf
import argparse
import random
import tensorboardX
from skimage.measure import compare_psnr

from utils import *
from dpdn_model import *

data_path = './dataset/'


def train_denoiser(max_iter=500000, batch_size=32, learning_rate=1e-4):
    val_set = np.load(data_path + 'val_SIDD_noisy.npy').astype(np.float32)
    val_gt = np.load(data_path + 'val_SIDD_gt.npy')
    val_num = val_set.shape[0]

    image_, gt_ = load_tfrecords(data_path + 'SIDD256.tfrecords', 2000, batch_size)

    image = tf.placeholder(tf.float32, [None, None, None, 3])
    gt = tf.placeholder(tf.float32, [None, None, None, 3])
    lr = tf.placeholder(tf.float32, None)

    output = DPDN(image, opt.C, opt.G, opt.D)

    t_vars = tf.trainable_variables()
    d_var = [var for var in t_vars if 'DPDN' in var.name]

    d_loss = tf.losses.absolute_difference(gt, output)

    d_opt = tf.train.AdamOptimizer(lr).minimize(d_loss,var_list=d_var)

    writer = tensorboardX.SummaryWriter('./logs_dn/' + opt.name)
    max_psnr = 0.0
    decay_rate = 0.1
    decapy_iter = [300000, 400000]

    get_paramsnum()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=d_var)
        for i in range(max_iter):
            if i in decapy_iter:
                learning_rate = learning_rate * decay_rate

            clean, noisy_s = sess.run([gt_, image_])
            _, loss_out, pred = sess.run([d_opt, d_loss, output],
                                         feed_dict={image: noisy_s,
                                                    gt: clean,
                                                    lr: learning_rate})
            if i % 100 == 0:
                out = np.clip(pred, 0.0, 1.0)
                psnr = batch_PSNR(out, clean, 1.0)
                print('D: Step {} loss : {}, psnr : {}'.format(i, loss_out, psnr))
                writer.add_scalar('D/step loss', loss_out, i)
                writer.add_scalar('D/step psnr', psnr, i)

            # Validation
            if i % 1000 == 0:
                psnr_val = 0.0
                loss_val = 0.0
                batchnum = val_num//batch_size
                for j in range(batchnum):
                    pred, loss_ = sess.run([output, d_loss], feed_dict={image: val_set[j*batch_size:(j + 1)*batch_size],
                                                                     gt: val_gt[j*batch_size:(j + 1)*batch_size]})
                    out = np.clip(pred, 0.0, 1.0)
                    psnr_val += batch_PSNR(out,val_gt[j*batch_size:(j + 1)*batch_size],1.0)
                    loss_val += loss_
                psnr_val = psnr_val / batchnum
                loss_val = loss_val / batchnum

                print('Validation loss : {} , PSNR : {}'.format(loss_val, psnr_val))
                writer.add_scalar('Val/loss', loss_val, i)
                writer.add_scalar('Val/psnr', psnr_val, i)
                if psnr_val > max_psnr:
                    saver.save(sess, './ckpt_dn/' + opt.name + '/dn_' + str(i) + '_' + str(psnr_val) + '.ckpt')
                    max_psnr = psnr_val
        saver.save(sess, './ckpt_dn/' + opt.name + '/cbd_last.ckpt')


def train_denoiser2(max_iter=500000, batch_size=32, learning_rate=1e-4):
    val_set = np.load(data_path + 'val_SIDD_noisy.npy').astype(np.float32)
    val_gt = np.load(data_path + 'val_SIDD_gt.npy')
    val_num = val_set.shape[0]

    image_, gt_ = load_tfrecords(data_path + 'data256shuffle.tfrecords', 2000, batch_size)
    image_2, gt_2, sigma_ = load_tfrecords_2(data_path + 'data256_CBD.tfrecords', 1000, batch_size)

    image = tf.placeholder(tf.float32, [None, None, None, 3])
    gt = tf.placeholder(tf.float32, [None, None, None, 3])
    lr = tf.placeholder(tf.float32, None)

    output = DPDN(image, opt.C, opt.G, opt.D)

    t_vars = tf.trainable_variables()
    d_var = [var for var in t_vars if 'DPDN' in var.name]

    d_loss = tf.losses.absolute_difference(gt, output)

    d_opt = tf.train.AdamOptimizer(lr).minimize(d_loss,var_list=d_var)

    writer = tensorboardX.SummaryWriter('./logs_dn/' + opt.name)
    max_psnr = 0.0
    decay_rate = 0.1
    decapy_iter = [300000, 400000]
    get_paramsnum()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(max_iter):
            if i in decapy_iter:
                learning_rate = learning_rate * decay_rate
            clean, noisy_s = sess.run([gt_2, image_2])
            _, loss_out, pred = sess.run([d_opt, d_loss, output],
                                         feed_dict={image: noisy_s,
                                                    gt: clean,
                                                    lr: learning_rate})
            clean, noisy_s = sess.run([gt_, image_])
            _, loss_out, pred = sess.run([d_opt, d_loss, output],
                                         feed_dict={image: noisy_s,
                                                    gt: clean,
                                                    lr: learning_rate})
            if i % 100 == 0:
                out = np.clip(pred, 0.0, 1.0)
                psnr = batch_PSNR(out, clean, 1.0)
                print('D: Step {} loss : {}, psnr : {}'.format(i, loss_out, psnr))
                writer.add_scalar('D/step loss', loss_out, i)
                writer.add_scalar('D/step psnr', psnr, i)

            # Validation
            if i % 1000 == 0:
                psnr_val = 0.0
                loss_val = 0.0
                batchnum = val_num//batch_size
                for j in range(batchnum):
                    pred, loss_ = sess.run([output, d_loss], feed_dict={image: val_set[j*batch_size:(j + 1)*batch_size],
                                                                     gt: val_gt[j*batch_size:(j + 1)*batch_size]})
                    out = np.clip(pred, 0.0, 1.0)
                    psnr_val += batch_PSNR(out,val_gt[j*batch_size:(j + 1)*batch_size],1.0)
                    loss_val += loss_
                psnr_val = psnr_val / batchnum
                loss_val = loss_val / batchnum

                print('Validation loss : {} , PSNR : {}'.format(loss_val, psnr_val))
                writer.add_scalar('Val/loss', loss_val, i)
                writer.add_scalar('Val/psnr', psnr_val, i)
                if psnr_val > max_psnr:
                    saver.save(sess, './ckpt_dn/' + opt.name + '/dn_' + str(i) + '_' + str(psnr_val) + '.ckpt')
                    max_psnr = psnr_val
        saver.save(sess, './ckpt_dn/' + opt.name + '/cbd_last.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DPDN")
    parser.add_argument("--batchsize", type=int, default=8, help="Training batch size")
    parser.add_argument("--maxiter", type=int, default=500000, help="Number of training epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--name", type=str, default='temp', help='name of experiments')
    parser.add_argument("--C", type=int, default=64, help='channel')
    parser.add_argument("--G", type=int, default=16, help='growth rate')
    parser.add_argument("--D", type=int, default=8, help='depth')
    parser.add_argument("--comment", type=str, help='description')
    opt = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    if not os.path.exists('./ckpt_dn/'+opt.name):
        os.mkdir('./ckpt_dn/'+opt.name)
    if not os.path.exists('./logs_dn/' + opt.name):
        os.mkdir('./logs_dn/' + opt.name )

    with open('./logs_dn/'+opt.name+'/model.txt','w') as f:
        f.write(str(opt._get_kwargs()))
    train_denoiser(opt.maxiter,opt.batchsize,opt.lr)