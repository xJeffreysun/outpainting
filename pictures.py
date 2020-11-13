#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:55:04 2020

@author: jeffreysun
"""

import os

import cv2
import numpy as np

# img = cv2.imread('/Users/jeffreysun/Desktop/tester.jpg')
# cv2.imshow('Original', img)
# img_half = cv2.resize(img, (128, 128))
# img_half_rgb = cv2.cvtColor(img_half, cv2.COLOR_BGR2RGB)

img_list = list()
directory = r'/Users/jeffreysun/Desktop/Outpainting/val_256'
count = 0
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        total_filename = directory + r'/' + filename
        img = cv2.imread(total_filename)
        img_half = cv2.resize(img, (128, 128))
        img_half_rgb = cv2.cvtColor(img_half, cv2.COLOR_BGR2RGB)
        rgb_norm = img_half_rgb / 255
        img_list.append(rgb_norm)
    if count == 10000:
        break
        # img_list.append(filename)

sliced_list = img_list[:10000]
sliced_array = np.array(sliced_list)
np.save(r'/Users/jeffreysun/Desktop/Outpainting/all_images.npy', np.array(img_list))

idx_test = np.random.choice(10000, 100, replace=False)
idx_train = list(set(range(10000)) - set(idx_test))
imgs_train = sliced_array[idx_train]
imgs_test = sliced_array[idx_test]
np.savez('places_128.npz', imgs_train=imgs_train, imgs_test=imgs_test, idx_train=idx_train, idx_test=idx_test)
