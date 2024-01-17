#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib to crop and resize an image.
"""

import sys
import math
import logging
import numpy as np
import pandas as pd
from skimage.transform import resize

def adjust_crop_and_resize_image(image_in, start_factor=[0.5,0.5], crop_factor=[1.0,1.0], l_target_dimension_px=[100,100]):

    inShapePx = image_in.shape
    crop_factor_half = [i * 0.5 for i in crop_factor]

    x_dimension_px      = math.floor(inShapePx[0] * crop_factor[0])
    x_dimension_half_px = math.floor(inShapePx[0] * crop_factor_half[0])

    x_start_px = math.floor(inShapePx[0] * start_factor[0]) - x_dimension_half_px
    x_end_px   = x_start_px + x_dimension_px

    y_dimension_px      = math.floor(inShapePx[1] * crop_factor[1])
    y_dimension_half_px = math.floor(inShapePx[1] * crop_factor_half[1])

    y_start_px = math.floor(inShapePx[1] * start_factor[1]) - y_dimension_half_px
    y_end_px   = y_start_px + y_dimension_px

    sLog = 'x: start/dimension/end('+str(x_start_px)+'/'+str(x_dimension_px)+'/'+str(x_end_px)+')'
    logging.info(sLog)
    sLog = 'y: start/dimension/end('+str(y_start_px)+'/'+str(y_dimension_px)+'/'+str(y_end_px)+')'
    logging.info(sLog)

    # sanity check target dimensions
    if start_factor[0]>1.0 or start_factor[1]>1.0:
        logging.error('start_factor >1 !  '+str(start_factor))
        sys.exit()
    if crop_factor[0]>1.0 or crop_factor[1]>1.0:
        logging.error('crop_factor >1 !  '+str(crop_factor))
        sys.exit()    

    image_cropped = image_in[x_start_px:x_end_px, y_start_px:y_end_px, :]
    image_out     = resize(image_cropped,(l_target_dimension_px[0],l_target_dimension_px[1],3))
    #print(image_out.shape)
    return image_out

'''
    start_pos  : x,y start position as factor of the original image size.
    crop_factor: x,y dimension factor of the original image size in range of 0...1.
'''
def image_crop_and_resize(df_in, start_factor=[0.5,0.5], crop_factor=[1.0,1.0], l_target_dimension_px=[100,100]):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()

    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        logging.info(' #'+str(cnt_row)+', '+str(start_factor)+', '+str(crop_factor)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
        
        image_adjusted = adjust_crop_and_resize_image(image_in, start_factor, crop_factor, l_target_dimension_px)
        nda_image_adjusted = image_adjusted
        logging.info(' #'+str(cnt_row)+', '+str(start_factor)+', '+str(crop_factor)+', '+str(target_int)+', '+str(nda_image_adjusted.shape)+', '+str(type(nda_image_adjusted)))
        
        dict2 = {'image': [nda_image_adjusted],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'cropresize_'+str(crop_factor)
                }
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])

        cnt_row = cnt_row + 1

    return df_out
