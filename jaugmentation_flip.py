#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib to flip images.
"""
import logging
import pandas as pd
import cv2
from plotly.subplots import make_subplots

def flip_image(image_in):
    image_out = cv2.flip(image_in, 1)
    return image_out

def image_flip(df_in):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()
    
    fig = make_subplots(rows=count_row, cols=2)
    
    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        logging.info(' #'+str(cnt_row)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
       
        image_flipped = flip_image(image_in)
        logging.info(' #'+str(cnt_row)+', '+str(target_int)+', '+str(image_flipped.shape)+', '+str(type(image_flipped)))
        
        dict2 = {'image': [image_flipped],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'flip'
                }
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])
       
        cnt_row = cnt_row + 1
    
    return df_out

