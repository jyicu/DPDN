import os
import numpy as np
import tensorflow as tf
import argparse
from skimage.measure import compare_psnr
import skimage.io as io

from dpdn_model import *


def argparser():
    parser = argparse.ArgumentParser(description='Dual Path Denoising Network')
    parser.add_argument('--imagepath', type=str, help='Path of test images',required=True)
    parser.add_argument('--model', type=str, default='./models/DPDN', help='Model path')
    parser.add_argument('--add_noise', action='store_true', help='Add Gaussian noise to test images')
    parser.add_argument('--sigma', type=float, default=25, help='std of added noise')
    parser.add_argument('--savepath', type=str, default='./results/', help='Path of denoised images')
    args = parser.parse_args()

    return args

def read_image(path):
    img = io.imread(path)
    img = img / 255.0
    return img.astype(np.float32)

def im2uint8(img):
    img = np.clip(img.squeeze(), 0.0, 1.0)
    img = (255*img).astype(np.uint8)
    return img

def test_images(args):
    imlist = os.listdir(args.imagepath)

    image = tf.placeholder(tf.float32, [None, None, None, 3])
    output = DPDN(image, 64, 16, 8)

    save_path = args.savepath
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,args.model)

        for i in range(len(imlist)):
            img = read_image(os.path.join(args.imagepath,imlist[i]))
            H, W, C = img.shape

            if args.add_noise:
                noisy = img + np.random.normal(0.0, (args.sigma / 255.0), img.shape)
                gt = img
                io.imsave(os.path.join(save_path, imlist[i].split('.')[0] +'_noisy.png'), im2uint8(noisy))
            else:
                noisy = img

            if H % 2 != 0:
                noisy = np.pad(noisy, ((0, 1), (0, 0), (0, 0)), 'reflect')
            if W % 2 != 0:
                noisy = np.pad(noisy, ((0, 0), (0, 1), (0, 0)), 'reflect')

            noisy = np.expand_dims(noisy,0)
            pred = sess.run(output, feed_dict={image: noisy})
            out = np.clip(pred, 0.0, 1.0)

            if H % 2 != 0:
                out = out[:,:-1,:,:]
            if W % 2 != 0:
                out = out[:,:,:-1,:]

            io.imsave(os.path.join(save_path, imlist[i].split('.')[0] + '_denoised.png'), im2uint8(out))


if __name__ == '__main__':
    args = argparser()
    test_images(args)
