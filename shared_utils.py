#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:17:36 2020

@author: andrea
"""
import tensorflow as tf

def gaussian_noise_layer(dim,std):
    '''generate noise mask of given dimension'''
    noise = tf.random.normal(shape=[dim, dim, 1], mean=0.0, stddev=std, dtype=tf.float32) / 255.0
    return noise



def add_noise(image,dim,NOISE_FORM,NOISE_STD):
    
    noise = gaussian_noise_layer(dim,NOISE_STD)
    if NOISE_FORM == 'GAUSS':
        noisy_image = tf.clip_by_value(image + noise, 0, 1)
    elif NOISE_FORM == 'RICE' :
        noisy_image = tf.clip_by_value(tf.math.sqrt(tf.math.square(image) + tf.math.square(noise)), 0, 1)
   
    return noisy_image