import tensorflow as tf
import glob
import numpy as np
import skimage.measure


def _parse_function_CBD(example_proto):
    keys_to_features = {'Noisy': tf.FixedLenFeature([], tf.string),
                        'GT': tf.FixedLenFeature([], tf.string),
                        'Sigma': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    noisy = parsed_features['Noisy']
    noisy = tf.decode_raw(noisy,  tf.float32)
    noisy = tf.reshape(noisy, [256, 256, 3])

    gt = parsed_features['GT']
    gt = tf.decode_raw(gt,  tf.float32)
    gt = tf.reshape(gt, [256, 256, 3])

    sigma = parsed_features['Sigma']
    sigma = tf.decode_raw(sigma,  tf.float32)
    sigma = tf.reshape(sigma, [256, 256, 3])

    return noisy, gt, sigma

def load_tfrecords_2(tfrecords_file, n_shuffle=1000, batch_size=64):
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset.map(_parse_function_CBD)

    dataset = dataset.shuffle(n_shuffle)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    x, y, z = iterator.get_next()
    return x, y, z

def _parse_function(example_proto):
    keys_to_features = {'Noisy': tf.FixedLenFeature([], tf.string),
                        'GT': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    noisy = parsed_features['Noisy']
    noisy = tf.divide(tf.cast(tf.decode_raw(noisy, tf.uint8), tf.float32), 255.)
    noisy = tf.reshape(noisy, [256, 256, 3])

    gt = parsed_features['GT']
    gt = tf.divide(tf.cast(tf.decode_raw(gt, tf.uint8), tf.float32), 255.)
    gt = tf.reshape(gt, [256, 256, 3])
    return noisy, gt

def load_tfrecords(tfrecords_file, n_shuffle=1000, batch_size=64):
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(n_shuffle)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    return x, y


def tf_psnr(pred, ref):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=ref * 255.0, predictions=pred * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def batch_PSNR(noisy, ref, data_range):
    PSNR = 0.0
    for i in range(noisy.shape[0]):
        PSNR += skimage.measure.compare_psnr(ref[i, :, :, :], noisy[i, :, :, :], data_range=data_range)
    return (PSNR / noisy.shape[0])


def dataaugment(patches):
    for i in range(patches.shape[0]):
        patches[i] = datatransform(patches[i], np.random.randint(0, 8))
    return patches


def dataaugment_idx(patches, idx):
    for i in range(patches.shape[0]):
        patches[i] = datatransform(patches[i], idx[i])
    return patches


def datatransform(img, mode):
    if mode < 4:
        img = np.rot90(img, 1)
    mode = mode % 4
    if mode == 0:
        pass
    if mode == 1:
        img = np.fliplr(img)
    if mode == 2:
        img = np.flipud(img)
    if mode == 3:
        img = np.flipud(np.fliplr(img))
    return img

def datatransform_inv(img, mode):
    mode_ = mode % 4
    if mode_ == 0:
        pass
    if mode_ == 1:
        img = np.fliplr(img)
    if mode_ == 2:
        img = np.flipud(img)
    if mode_ == 3:
        img = np.fliplr(np.flipud(img))
    if mode < 4:
        img = np.rot90(img, -1)
    return img


def write_description(opt):
    with open('./logs_da/' + opt.name + '/model.txt', 'w') as f:
        f.write(str(opt._get_kwargs()))


def get_paramsnum():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)
    return

def batch_PSNR_255(noisy, ref):
    PSNR = 0.0
    for i in range(noisy.shape[0]):
        ref_i = np.round(255*ref[i, :, :, :]).astype(np.uint8)
        noisy_i = np.round(255*noisy[i, :, :, :]).astype(np.uint8)
        PSNR += skimage.measure.compare_psnr(ref_i, noisy_i, data_range=255)
    return (PSNR / noisy.shape[0])
