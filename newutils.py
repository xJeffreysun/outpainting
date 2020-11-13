#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:16:40 2020

@author: jeffreysun
"""

import numpy as np
from PIL import Image
import cv2
import os
import re
import imageio

IMAGE_SZ = 128

dir_PATH = '/Users/jeffreysun/Desktop/Outpainting/resized'

resize('/Users/jeffreysun/Desktop/Outpainting/val_256/Places365_val_00000177.jpg', '/Users/jeffreysun/Desktop/Outpainting/resized/test.jpg')
save_image(tester_imgs[0], '/Users/jeffreysun/Desktop/Outpainting/show.jpg')
processed_imgs = preprocess_images_outpainting(tester_imgs)
cropped_img = processed_imgs[:, :, :, :3]
masked_img = processed_imgs[:, :, :, 3]
save_image(cropped_img[0], '/Users/jeffreysun/Desktop/Outpainting/show.jpg')

def resize(in_PATH, out_PATH):
    img = Image.open(in_PATH).convert('RGB')
    img_scale = img.resize((IMAGE_SZ, IMAGE_SZ), Image.ANTIALIAS)
    img_scale.save(out_PATH, format='JPG')

def load_images(in_PATH, verbose=False):
    # in_PATH is a path from cwd
    # verbose prints steps in the command line
    # empty array
    imgs =[]
    # listdir returns a list containing the names of the files in the directory
    for filename in sorted(os.listdir(in_PATH)[:100]):
        if verbose:
            print('Processing %s' % filename)
        # gets absolute path of each image
        full_filename = os.path.join(os.path.abspath(in_PATH), filename)
        # opens image with PIL and as RGB format
        img = Image.open(full_filename).convert('RGB')
        # converts the PIL object to an array
        pix = np.array(img)
        # normalizes to [0,1]
        pix_norm = pix / 255.0
        # adds to list of image arrays
        imgs.append(pix_norm)
    return np.array(imgs)

def compile_images(in_PATH, out_PATH):
    # imgs an array of image arrays
    imgs = load_images(in_PATH, verbose=True)
    # saves the array as a .npy file
    np.save(out_PATH, imgs)

def preprocess_images_outpainting(imgs, crop=True):
    # saves the length of the first dimension of the imgs array
    # i.e. the number of images
    m = imgs.shape[0]
    # makes sure they are numpy arrays
    imgs = np.array(imgs, copy=True)
    # calculates the average [0,1] pixel value
    # for the left and right part of each image
    # i.e. the part that is unmasked
    # and then takes their average
    left_pix_avg = np.mean(imgs[:, :, :int(2 * IMAGE_SZ / 8), :], axis=(1, 2, 3))
    right_pix_avg = np.mean(imgs[:, :, int(-2 * IMAGE_SZ / 8):, :], axis=(1, 2, 3))
    pix_avg = np.mean(np.array([left_pix_avg, right_pix_avg]), axis=0)
    # pix_avg = np.mean(imgs, axis=(1, 2, 3))
    if crop:
        # sets the left and right part of each image
        # to the average pixel density
        imgs[:, :, :int(2 * IMAGE_SZ / 8), :] = imgs[:, :, int(-2 * IMAGE_SZ / 8):, :] = pix_avg[:, np.newaxis, np.newaxis, np.newaxis]
    # sets the mask to hold all zeros, meaning black
    mask = np.zeros((m, IMAGE_SZ, IMAGE_SZ, 1))
    # sets the left and right of the mask to 1, meaning white
    mask[:, :, :int(2 * IMAGE_SZ / 8), :] = mask[:, :, int(-2 * IMAGE_SZ / 8):, :] = 1.0
    # concatenates the 3 rgb layers with a mask layer for a 
    # m, 128, 128, 4 shape array
    imgs_p = np.concatenate((imgs, mask), axis=3)
    return imgs_p

def norm_image(img_r):
    img_norm = (img_r * 255.0).astype(np.uint8)
    return img_norm

def vis_image(img_r, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.show()

def save_image(img_r, name, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.save(name, format='PNG')