'''
This script makes a hdf5 style dataset with all images in a chosen directory.
Gram matrices computed here are never normalized by the number of channels.
Normalization is done if necessary on the training stage.
'''
import numpy as np
import h5py

import keras
import keras.backend as K
from keras.applications import vgg16

from training import get_style_features
from utils import preprocess_image_scale, config_gpu, std_input_list

import os
import argparse

if __name__ == "__main__":

    def_sl = ['block1_conv2', 'block2_conv2',
              'block3_conv3', 'block4_conv3']

    parser = argparse.ArgumentParser()
    parser.add_argument('--style_dir', type=str, default='gram_imgs',
                        help='Directory that contains the images.')
    parser.add_argument('--gram_dataset_path', type=str, default='grams.h5',
                        help='Name of the output hdf5 file.')
    parser.add_argument('--style_imgs', type=str, default=None, nargs='+',
                        help='Style image file names.')
    parser.add_argument('--style_img_size', type=int, default=[None], nargs='+',
                        help='Largest size of the style images')
    parser.add_argument('--style_layers', type=str, nargs='+', default=def_sl)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')
    args = parser.parse_args()

    config_gpu(args.gpu, args.allow_growth)

    loss_net = vgg16.VGG16(weights='imagenet', include_top=False)

    targets_dict = dict([(layer.name, layer.output) for layer in loss_net.layers])

    s_targets = get_style_features(targets_dict, args.style_layers)

    get_style_target = K.function([loss_net.input], s_targets)
    gm_lists = [[] for l in args.style_layers]

    img_list = []
    img_size_list = []
    # Get style image names or get all images in the directory
    if args.style_imgs is None:
        args.style_imgs = os.listdir(args.style_dir)

    # Check the image sizes
    args.style_img_size = std_input_list(args.style_img_size, len(args.style_imgs), 'Image size')

    for img_name, img_size in zip(args.style_imgs, args.style_img_size):
        try:
            print(img_name)
            img = preprocess_image_scale(os.path.join(args.style_dir, img_name),
                                         img_size=img_size)
            s_targets = get_style_target([img])
            for l, t in zip(gm_lists, s_targets):
                l.append(t)
            img_list.append(os.path.splitext(img_name)[0])
            img_size_list.append(img_size)
        except IOError as e:
            print('Could not open file %s as image.' %img_name)

    mtx = []
    for l in gm_lists:
        mtx.append(np.concatenate(l))

    f = h5py.File(args.gram_dataset_path, 'w')

    f.attrs['img_names'] = img_list
    f.attrs['img_sizes'] = img_size_list
    for name, m in zip(args.style_layers, mtx):
        f.create_dataset(name, data=m)

    f.flush()
    f.close()
