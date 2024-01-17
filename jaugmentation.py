#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Augmentation lib.
"""
import logging
import pandas as pd

import jaugmentation_flip       as jaflip
import jaugmentation_rotation   as jarotation
import jaugmentation_brightness as jabrightness
import jaugmentation_saturation as jasaturation
import jaugmentation_contrast   as jacontrast
import jaugmentation_noise      as janoise
import jaugmentation_cropresize as jacropresize
import jplot

'''
    Augments original input images.
    Each augmentation method is applied to each original input image.
'''
def image_augmentation_individual(df_in):
    logging.info('START')

    df_augmented = pd.DataFrame()
    
    df_augmented = pd.concat([df_augmented, df_in])

    df_images_flipped = jaflip.image_flip(df_in)
    df_augmented = pd.concat([df_augmented, df_images_flipped])

    df_images_rotated = jarotation.image_rotation(df_in, 10)
    df_augmented = pd.concat([df_augmented, df_images_rotated])
    df_images_rotated = jarotation.image_rotation(df_in, 20)
    df_augmented = pd.concat([df_augmented, df_images_rotated])
    df_images_rotated = jarotation.image_rotation(df_in, 30)
    df_augmented = pd.concat([df_augmented, df_images_rotated])
    
    df_images_rotated = jarotation.image_rotation(df_in, -10)
    df_augmented = pd.concat([df_augmented, df_images_rotated])
    df_images_rotated = jarotation.image_rotation(df_in, -20)
    df_augmented = pd.concat([df_augmented, df_images_rotated])
    df_images_rotated = jarotation.image_rotation(df_in, -30)
    df_augmented = pd.concat([df_augmented, df_images_rotated])
    
    df_images_brightness = jabrightness.image_brightness(df_in, 0.02)
    df_augmented = pd.concat([df_augmented, df_images_brightness])
    df_images_brightness = jabrightness.image_brightness(df_in, 0.10)
    df_augmented = pd.concat([df_augmented, df_images_brightness])

    df_images_brightness = jabrightness.image_brightness(df_in, -0.02)
    df_augmented = pd.concat([df_augmented, df_images_brightness])
    df_images_brightness = jabrightness.image_brightness(df_in, -0.10)
    df_augmented = pd.concat([df_augmented, df_images_brightness])
    
    df_images_noise = janoise.image_noise(df_in, 'gaussian')
    df_augmented = pd.concat([df_augmented, df_images_noise])
    df_images_noise = janoise.image_noise(df_in, 's&p', 0.1)
    df_augmented = pd.concat([df_augmented, df_images_noise])

    df_images_saturated = jasaturation.image_saturation(df_in, 1)
    df_augmented = pd.concat([df_augmented, df_images_saturated])
    df_images_saturated = jasaturation.image_saturation(df_in, 10)
    df_augmented = pd.concat([df_augmented, df_images_saturated])
    df_images_saturated = jasaturation.image_saturation(df_in, 20)
    df_augmented = pd.concat([df_augmented, df_images_saturated])

    for contrast_factor in [0.1, 0.2, 0.3, 0.6, 0.9, 1.3, 1.5, 1.8, 2.0]:
        df_images_contrast = jacontrast.image_contrast(df_in, contrast_factor)
        df_augmented = pd.concat([df_augmented, df_images_contrast])

    for crop_factor in [ [0.2,0.2], [0.4,0.4], [0.6,0.6], [0.8,0.8]]:
        df_images_cropped_and_resized = jacropresize.image_crop_and_resize(df_in, [0.5,0.5], crop_factor)
        df_augmented = pd.concat([df_augmented, df_images_cropped_and_resized])

    jplot.plot_df(df_augmented)

    #logging.info('df_augmented: ('+str(df_augmented.shape)+')\n'+str(df_augmented))
    #logging.info('df_augmented: ('+str(df_augmented.shape))
    logging.info(' ... END')

    return df_augmented

'''
    Augments original input images.
    Augmentation methods are combined in order to 
    generate a wealth of sample variations.
'''
def image_augmentation_combined(df_in, l_target_dimension_px):
    logging.info('START')

    df_augmented = pd.DataFrame()
    
    df_augmented = pd.concat([df_augmented, df_in])

    df_images_flipped = jaflip.image_flip(df_in)
    df_augmented = pd.concat([df_augmented, df_images_flipped])

    df_working = pd.DataFrame()
    #for delta_brightness in [ -0.1, -0.05, -0.02, 0.02, 0.05, 0.1 ]:
    for delta_brightness in [ -0.1, 0.1 ]:
        df_images_brightness = jabrightness.image_brightness(df_augmented, delta_brightness)
        df_working           = pd.concat([df_working, df_images_brightness])
    df_augmented = pd.concat([df_augmented, df_working])

    df_working = pd.DataFrame()
    for saturation_factor in [ 20 ]:
        df_images_saturated = jasaturation.image_saturation(df_augmented, saturation_factor)
        df_working          = pd.concat([df_working, df_images_saturated])
    df_augmented = pd.concat([df_augmented, df_working])

    df_working = pd.DataFrame()
    #for contrast_factor in [0.1, 0.2, 0.3, 0.6, 0.9, 1.3, 1.5, 1.8, 2.0]:
    for contrast_factor in [ 0.9 ]:
        df_images_contrast = jacontrast.image_contrast(df_augmented, contrast_factor)
        df_working         = pd.concat([df_working, df_images_contrast])
    df_augmented = pd.concat([df_augmented, df_working])

    df_working = pd.DataFrame()
    for lSettings in [ ['gaussian',None], ['s&p',0.1] ]:
        df_images_noise = janoise.image_noise(df_augmented, lSettings[0], lSettings[1])
        df_working      = pd.concat([df_working, df_images_noise])
    df_augmented = pd.concat([df_augmented, df_working])

    df_working = pd.DataFrame()
    for angle in [ -20, -10, -5, 5, 10, 20 ]:
        df_images_rotated = jarotation.image_rotation(df_augmented, angle)
        df_working        = pd.concat([df_working, df_images_rotated])
    df_augmented = pd.concat([df_augmented, df_working])

    df_working = pd.DataFrame()
    #for crop_factor in [ [0.2,0.2], [0.4,0.4], [0.6,0.6], [0.8,0.8] ]:
    for crop_factor in [ [0.4,0.4], [0.6,0.6] ]:
        df_images_cropped_and_resized = jacropresize.image_crop_and_resize(df_augmented, [0.5,0.5], crop_factor, l_target_dimension_px)
        df_working                    = pd.concat([df_working, df_images_cropped_and_resized])
    df_augmented = pd.concat([df_augmented, df_working])

    jplot.plot_df(df_augmented)

    logging.info(' ... END')
    return df_augmented
