#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib to adjust image contrast.
"""

import logging
import numpy as np
import pandas as pd

import tensorflow as tf

def adjust_contrast_image(image_in, contrast_factor=0):
    """
    Adjusts image contrast
        contrast_factor should be in the range (-inf, inf)
    """
    image_out = tf.image.adjust_contrast(image_in, 
        contrast_factor=contrast_factor)
    
    return image_out

def image_contrast(df_in, contrast_factor=0.1):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()

    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        logging.info(' #'+str(cnt_row)+', '+str(contrast_factor)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
        
        image_contrast = adjust_contrast_image(image_in, contrast_factor)
        nda_image_contrast = image_contrast.numpy()
        logging.info(' #'+str(cnt_row)+', '+str(contrast_factor)+', '+str(target_int)+', '+str(nda_image_contrast.shape)+', '+str(type(nda_image_contrast)))
        
        dict2 = {'image': [nda_image_contrast],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'contrast_'+str(contrast_factor)}
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])

        cnt_row = cnt_row + 1

    return df_out
