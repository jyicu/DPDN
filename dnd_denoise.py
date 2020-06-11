 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

import numpy as np
import scipy.io as sio
import os
import h5py
import argparse
from dpdn_model import *

def bundle_submissions_srgb(submission_folder):
     '''
     Bundles submission data for sRGB denoising

     submission_folder Folder where denoised images reside

     Output is written to <submission_folder>/bundled/. Please submit
     the content of this folder.
     '''
     out_folder = os.path.join(submission_folder, "bundled/")
     try:
         os.mkdir(out_folder)
     except:
         pass
     israw = False
     eval_version = "1.0"

     for i in range(50):
         Idenoised = np.zeros((20,), dtype=np.object)
         for bb in range(20):
             filename = '%04d_%02d.mat' % (i + 1, bb + 1)
             s = sio.loadmat(os.path.join(submission_folder, filename))
             Idenoised_crop = s["Idenoised_crop"]
             Idenoised[bb] = Idenoised_crop
         filename = '%04d.mat' % (i + 1)
         sio.savemat(os.path.join(out_folder, filename),
                     {"Idenoised": Idenoised,
                      "israw": israw,
                      "eval_version": eval_version},
                     )

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma

def denoise_srgb(denoiser, data_folder, out_folder, args):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:
        pass

    image = tf.placeholder(tf.float32, [None, None, None, 3])
    output = DPDN(image,64,16,8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print('model loaded\n')
        # load info
        infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
        info = infos['info']
        bb = info['boundingboxes']
        print('info loaded\n')
        # process data
        for i in range(50):
            filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
            img = h5py.File(filename, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)
            # bounding box
            ref = bb[0][i]
            boxes = np.array(info[ref]).T
            for k in range(20):
                idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
                Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
                H = Inoisy_crop.shape[0]
                W = Inoisy_crop.shape[1]
                nlf = load_nlf(info, i)
                # for yy in range(2):
                #     for xx in range(2):
                nlf["sigma"] = load_sigma_srgb(info, i, k)
                Idenoised_crop = denoiser(sess,image,output,Inoisy_crop, nlf)
                # save denoised data
                Idenoised_crop = np.float32(Idenoised_crop)
                save_file = os.path.join(out_folder, '%04d_%02d.mat' % (i + 1, k + 1))
                sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
                print('%s crop %d/%d' % (filename, k + 1, 20))
            print('[%d/%d] %s done\n' % (i + 1, 50, filename))

def test_dnd_with_sess(sess, image, output, patch, nlf):
    img = np.expand_dims(patch,0)
    pred = sess.run(output, feed_dict={image: img})
    pred = np.clip(pred, 0, 1).squeeze()
    return pred

def argparser():
    parser = argparse.ArgumentParser(description='Dual Path Denoising Network')
    parser.add_argument('--datafolder', type=str, required=True, help='Folder where the DND dataset resides')
    parser.add_argument('--outfolder', type=str, required=True, help='Folder where denoised output should be written to')
    parser.add_argument('--model', type=str, default='./models/DPDN', help='Path of Model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    denoise_srgb(test_dnd_with_sess, args.datafolder, args.outfolder,args)
    bundle_submissions_srgb(args.outfolder)
