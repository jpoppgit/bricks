#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib to adjust image brightness.
"""

import logging
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

import tensorflow as tf

def adjust_brightness_image(image_in, delta=0):
    """
    Adjusts image brightness
        delta should be in the range (-1,1)
        -1: Will decrease brightness (towards black).
        +1: Will increase brightness (towards white).
    """
    image_out = tf.image.adjust_brightness(image_in, delta=delta)
    
    #seed = (0, 0) # tuple of size (2,)
    #image_out = tf.image.stateless_random_brightness(
    #  image_in, max_delta=0.3, seed=seed)
    
    return image_out

def image_brightness(df_in, delta_brightness=0.1):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()
    
    fig = make_subplots(rows=count_row, cols=2)
    
    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        logging.info(' #'+str(cnt_row)+', '+str(delta_brightness)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
        
        image_brightness     = adjust_brightness_image(image_in, delta_brightness)
        nda_image_brightness = image_brightness.numpy()
        logging.info(' #'+str(cnt_row)+', '+str(delta_brightness)+', '+str(target_int)+', '+str(nda_image_brightness.shape)+', '+str(type(nda_image_brightness)))
        
        dict2 = {'image': [nda_image_brightness],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'brightness_'+str(delta_brightness)
                }
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])
        
        cnt_row = cnt_row + 1
    
    return df_out
