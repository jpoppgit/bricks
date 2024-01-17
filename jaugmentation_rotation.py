#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib to rotate images.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from plotly.subplots import make_subplots

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def image_rotation(df_in, angle=30):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()
    
    fig = make_subplots(rows=count_row, cols=2)
    
    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        logging.info(' #'+str(cnt_row)+', '+str(angle)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
       
        image_rotated = rotate_image(image_in, angle)
        logging.info(' #'+str(cnt_row)+', '+str(angle)+', '+str(target_int)+', '+str(image_rotated.shape)+', '+str(type(image_rotated)))        
        
        dict2 = {'image': [image_rotated],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'rotate_'+str(angle)}
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])
        
        cnt_row = cnt_row + 1

    return df_out
 