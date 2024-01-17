#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Helper lib.
"""
import os
import shutil
import math
import logging
import numpy as np

logging.basicConfig(format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
                    level=logging.INFO)

'''
    Re-creates a directory. Deletes the directory if it exists before.
'''
def recreateDir(sDir):
    if os.path.exists(sDir):
        shutil.rmtree(sDir)
    os.mkdir(sDir)

'''
    IN : Pandas series containing images in format (X,Y,3).
    OUT: List containing flattened images.
'''
def flatten_elements(pd_series_in):
    l_out = []
    logging.info('#'+str(len(pd_series_in)))
    for idx, el in enumerate(pd_series_in):
        if isinstance(el, str) or isinstance(el, int):
            # for labels as string
            #logging.info('#'+str(idx)+' '+str(type(el))+' '+str(el))
            l_out.append(el)
        else:
            # for images, e.g. in format (X,Y,3)
            #logging.info('#'+str(idx)+' '+str(type(el))+' '+str(el.shape))
            el_flattened = el.flatten()
            #logging.info('#'+str(idx)+' '+str(type(el_flattened))+' '+str(el_flattened.shape))
            l_out.append(el_flattened)
    return l_out

'''
    Find most frequent element in a list.
'''
def most_frequent(l_in):
    '''
    Make a set of the list so that the duplicate elements are deleted. 
    Then find the highest count of occurrences of each element in 
    the set and thus, we find the maximum out of it. 
    '''
    return max(set(l_in), key = l_in.count)

'''
    Find the next squared number to input 'n'.
    Also returns the root of the next squared number.
'''
def nextPerfectSquare(N):
    nextN = math.floor(math.sqrt(N)) + 1
    return [nextN, nextN * nextN]
