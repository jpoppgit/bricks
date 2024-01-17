#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib to adjust image saturation.
"""

import logging
import numpy as np
import pandas as pd

import tensorflow as tf

def adjust_saturation_image(image_in, saturation_factor=0):
    """
    Adjusts image saturation
        saturation_factor should be in the range [0, inf)
    """
    image_out = tf.image.adjust_saturation(image_in, 
        saturation_factor=saturation_factor)
    
    return image_out

def image_saturation(df_in, saturation_factor=0.1):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()

    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        logging.info(' #'+str(cnt_row)+', '+str(saturation_factor)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
        
        image_saturation = adjust_saturation_image(image_in, saturation_factor)
        nda_image_saturation = image_saturation.numpy()
        logging.info(' #'+str(cnt_row)+', '+str(saturation_factor)+', '+str(target_int)+', '+str(nda_image_saturation.shape)+', '+str(type(nda_image_saturation)))
        
        dict2 = {'image': [nda_image_saturation],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'saturation_'+str(saturation_factor)
                }
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])

        cnt_row = cnt_row + 1

    return df_out
