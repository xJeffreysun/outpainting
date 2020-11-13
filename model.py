#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:38:40 2020

@author: jeffreysun
"""

from tensorflow.keras import layers


def generator(z):
    conv1 = layers.Conv2D(
        inputs=z,
        filters=64, kernel_size=[5, 5],
        strides=(1, 1),
        padding='same',
        activation='relu'
    )

    conv2 = layers.Conv2D(
        inputs=conv1,
        filters=128,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding='same',
        activation='relu'
    )

    conv3 = layers.Conv2D(
        inputs=conv2,
        filters=256,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding='same',
        activation='relu'
    )

    conv4 = layers.Conv2D(
        inputs=conv3,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        dilation_rate=(2, 2),
        padding='same',
        activation='relu'
    )

    conv5 = layers.Conv2D(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        dilation_rate=(4, 4),
        padding='same',
        activation='relu'
    )

    conv5_p = layers.Conv2D(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        dilation_rate=(8, 8),
        padding='same',
        activation='relu'
    )

    conv6 = layers.Conv2D(
        inputs=conv5_p,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation='relu'
    )

    deconv7 = layers.Conv2DTranspose(
        inputs=conv6,
        filters=128,
        kernel_size=[4, 4],
        strides=(2, 2),
        padding='same',
        activation='relu'
    )

    conv8 = layers.Conv2D(
        inputs=deconv7,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation='relu'
    )

    out = layers.Conv2D(
        inputs=conv8,
        filters=3,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation='sigmoid'
    )

    return out


