#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation helper to apply noise to an image.
"""
import logging
import pandas as pd
import skimage

def noise_image(image_in, mode, amount):
    if mode == 'gaussian':
        image_out = skimage.util.random_noise(image_in, mode=mode)
    elif mode == 's&p':
        image_out = skimage.util.random_noise(image_in, mode=mode, amount=amount)
    else:
        logging.error(mode+' not supported!')
    return image_out

def image_noise(df_in, mode, amount=None):
    #logging.info(df_in.keys())
    count_row = df_in.shape[0]
    
    df_out = pd.DataFrame()

    cnt_row = 1
    for index, row in df_in.iterrows():
        image_in      = row['image']
        target_int    = row['target_int']
        sAugmentation = row['augmentation']
        #logging.info(' #'+str(cnt_row)+', '+str(target_int)+', '+str(image_in.shape)+', '+str(type(image_in)))
       
        image_processed = noise_image(image_in, mode, amount)
        logging.info(' #'+str(cnt_row)+', '+str(target_int)+', '+mode+', '+str(image_processed.shape)+', '+str(type(image_processed)))
        
        dict2 = {'image': [image_processed],
                 'target_int': target_int,
                 'augmentation': sAugmentation+'<br>'+'noise_'+mode+'-'+str(amount)}
        df2 = pd.DataFrame.from_records(dict2)
        df_out = pd.concat([df_out, df2])
        
        cnt_row = cnt_row + 1
    return df_out
