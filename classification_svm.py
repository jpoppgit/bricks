#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Classification using SVM.
"""

import os
import logging
import time
import argparse, textwrap
from datetime import datetime

import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import resize
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from scipy import stats

import tensorflow as tf
from tensorflow.keras.datasets import mnist

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

import jlogger # registers 'logging.basicConfig' once
import jplot
import jhelper
import jaugmentation

HERE_RELATIVE = os.path.relpath(os.path.dirname(__file__))

def load_base_images(PATH_BASE_IMAGES, l_base_categories, l_target_dimension_px):
    df_out = pd.DataFrame()

    for s_category in l_base_categories:
        logging.info('loading... category('+s_category+')')
        path = os.path.join(PATH_BASE_IMAGES, s_category)
        for img in os.listdir(path):
            image = imread(os.path.join(path, img))
            
            # resize to common shape
            image_resized = resize(image,(l_target_dimension_px[0],l_target_dimension_px[1],3))
            logging.info('resized: '+str(type(image_resized))+' '+str(image.shape)+' -> '+str(image_resized.shape))
    
            #image_flattened = image_resized.flatten()
            #logging.info('flattened: '+str(type(image_flattened))+', '+str(image_flattened.shape))

            # Add another row using the "dictionary way"
            dict2 = {'image': [image_resized], 'target': s_category, 'augmentation': 'base'}
            df2 = pd.DataFrame.from_records(dict2)
            df_out = pd.concat([df_out, df2])
             
        #logging.info('loaded category: '+s_category+' successfully')
    
    #logging.info('df_out: \n'+str(df_out))
    return df_out

'''
    Maps categorial string to integer value.
'''
def set_target_int(df_in, l_base_categories):
    df_out = df_in

    dictMap = {}
    for idx, s_base_category in enumerate(l_base_categories):
        dictMap[s_base_category] = int(idx)
    df_out['target_int'] = df_out['target'].map(dictMap)
    logging.info('\n'+str(df_out))

    # sanity check about availabe 'target_int' values
    logging.info('target_int:'+str(df_out['target_int'].unique()))

    return df_out


def jtrain_test_split(df):
    logging.info(df.keys())
    x = df['image']
    y = df['target_int']
    logging.info('#x('+str(len(x))+')'+' #y('+str(len(y))+')')
    pd_series_x_train, pd_series_x_test, \
    pd_series_y_train, pd_series_y_test = \
        train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

    x_train = jhelper.flatten_elements(pd_series_x_train)
    x_test  = jhelper.flatten_elements(pd_series_x_test)
    y_train = jhelper.flatten_elements(pd_series_y_train)
    y_test  = jhelper.flatten_elements(pd_series_y_test)

    sLog  = '\n #x_train('+str(len(x_train))+')'+'/'+str(x_train[0].shape)+' '+str(type(x_train))
    sLog += '\n #x_test('+str(len(x_test))+')'+'/'+str(x_test[0].shape)+' '+str(type(x_test))
    sLog += '\n #y_train('+str(len(y_train))+')'+'/'+str(y_train[0])+' '+str(type(y_train))
    sLog += '\n #y_test('+str(len(y_test))+')'+'/'+str(y_test[0])+' '+str(type(y_test))
    logging.info(sLog)
    
    return x_train, x_test, y_train, y_test


def process_bricks(args, DIR_OUTPUT):
    logging.info('sklearn version   : '+str(sklearn.__version__))
    logging.info('Tensorflow version: '+str(tf.__version__))
    logging.info('skimage version   : '+str(skimage.__version__))

    PATH_BASE_IMAGES = os.path.join(HERE_RELATIVE, 'data')
    l_base_categories     = ['blue_4_1', 'blue_6_1', 'blue_6_2']
    l_target_dimension_px = [100, 100]
    df_images_base = load_base_images(PATH_BASE_IMAGES, l_base_categories, l_target_dimension_px)
    df_images_base = set_target_int(df_images_base, l_base_categories)

    #df_images_augmented_individual = jaugmentation.image_augmentation_individual(df_images_base)
    #x_train, x_test, y_train, y_test = jtrain_test_split(df_images_augmented_individual)

    df_images_augmented_combined     = jaugmentation.image_augmentation_combined(df_images_base, l_target_dimension_px)
    x_train, x_test, y_train, y_test = jtrain_test_split(df_images_augmented_combined)

    # limit dataset
    if args.fast is True:
        x_train = x_train[0:50]
        y_train = y_train[0:50]
        x_test  = x_test[0:30]
        y_test  = y_test[0:30]

    jplot.plot_class_distribution('bricks', DIR_OUTPUT, l_base_categories, y_train, y_test)

    # With the help of GridSearchCV and parameters grid, create a model.
    # higher C    : higher error allowed
    # higher gamma: more fine granular 'hyperplane'; adapts more to data
    if args.fast is True:
        param_grid={'C':[0.1],'gamma':[0.0001],'kernel':['poly']}
    else:
        param_grid={'C':[0.1,10],
                    'gamma':[0.0001,10],
                    'kernel':['rbf','poly']}
        param_grid={'C':[0.1],'gamma':[0.0001],'kernel':['poly']}
    svc = svm.SVC(probability=True, decision_function_shape='ovo')
    model = GridSearchCV(svc, param_grid,
                         cv=2, n_jobs=-1, verbose=3)
    
    model.fit(x_train, y_train)
    # model.best_params_ contains the best parameters obtained from GridSearchCV
    logging.info('Used parameters: '+str(model.best_params_))
    logging.info('Used estimator : '+str(model.best_estimator_))
    logging.info('Classes        : '+str(model.classes_))
    logging.info('support_vectors: '+str(model.best_estimator_.support_vectors_.shape))

    # evaluation
    y_pred = model.predict(x_test)
    logging.info('y_pred: '+'#('+str(len(y_pred))+') '+str(y_pred))
    logging.info('y_test: '+'#('+str(len(y_test))+') '+str(y_test))
    logging.info('Accuracy('+str(accuracy_score(y_pred,y_test)*100)+'%)')

    # classification_report
    logging.info('\n'+str(classification_report(y_test, y_pred)))
    logging.info('Note \'precision\': The precision will be "how many are correctly classified among that class."')
    logging.info('Note \'recall\'   : The recall means "how many of this class you find over the whole number of element of this class."')
    logging.info('Note \'support\'  : The support is the number of occurence of the given class in your dataset.')

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    #logging.info('confusion matrix: '+str(cm))
    jplot.plot_confusion_matrix('bricks', cm, model.classes_, DIR_OUTPUT)

    jplot.svm_decision_multi('bricks', model,
                             x_train, y_train,
                             x_test,  y_test,
                             DIR_OUTPUT)

def process_mnist(args, DIR_OUTPUT):

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print('MNIST Dataset Shape:')
    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  ' + str(X_test.shape))
    print('Y_test:  ' + str(Y_test.shape))

    X_train_1d = []
    for idx in range(0, X_train.shape[0]):
        sample = X_train[idx]
        sample_1d = sample.reshape(-1) # 2d -> 1d array
        X_train_1d.append(sample_1d)
    X_train_1d = np.array(X_train_1d)
    print('X_train_1d: ' + str(X_train_1d.shape))

    Y_train_1d = []
    for idx in range(0, Y_train.shape[0]):
        sample = Y_train[idx]
        sample_1d = sample.reshape(-1) # 2d -> 1d array
        Y_train_1d.append(sample_1d)
    Y_train_1d = np.array(Y_train_1d)
    print('Y_train_1d: ' + str(Y_train_1d.shape))

    X_test_1d = []
    for idx in range(0, X_test.shape[0]):
        sample = X_test[idx]
        sample_1d = sample.reshape(-1) # 2d -> 1d array
        X_test_1d.append(sample_1d)
    X_test_1d = np.array(X_test_1d)
    print('X_test_1d: ' + str(X_test_1d.shape))

    Y_test_1d = []
    for idx in range(0, Y_test.shape[0]):
        sample = Y_test[idx]
        sample_1d = sample.reshape(-1) # 2d -> 1d array
        Y_test_1d.append(sample_1d)
    Y_test_1d = np.array(Y_test_1d)
    print('Y_test_1d: ' + str(Y_test_1d.shape))

    # 0,1,2,3 only
    tuple_y_train = np.where( (Y_train_1d == 0) | (Y_train_1d == 1) | (Y_train_1d == 2) | (Y_train_1d == 3) )
    tuple_y_train = np.asarray(tuple_y_train)
    idx_y_train = tuple_y_train[0]
    X_train_1d = X_train_1d[idx_y_train]
    Y_train_1d = Y_train_1d[idx_y_train]
    print('X_train_1d: ' + str(X_train_1d.shape))
    print('Y_train_1d: ' + str(Y_train_1d.shape))

    tuple_y_test = np.where( (Y_test_1d == 0) | (Y_test_1d == 1) | (Y_test_1d == 2) | (Y_test_1d == 3))
    tuple_y_test = np.asarray(tuple_y_test)
    idx_y_test = tuple_y_test[0]
    X_test_1d = X_test_1d[idx_y_test]
    Y_test_1d = Y_test_1d[idx_y_test]
    print('X_test_1d: ' + str(X_test_1d.shape))
    print('Y_test_1d: ' + str(Y_test_1d.shape))

    # limit dataset
    if args.fast is True:
        X_train_1d = X_train_1d[0:50,:]
        Y_train_1d = Y_train_1d[0:50,:]
        X_test_1d  = X_test_1d[0:30,:]
        Y_test_1d  = Y_test_1d[0:30,:]
    # else:
    #     X_train_1d = X_train_1d[0:10000,:]
    #     Y_train_1d = Y_train_1d[0:10000,:]
    #     X_test_1d  = X_test_1d[0:2000,:]
    #     Y_test_1d  = Y_test_1d[0:2000,:]

    logging.info('MNIST: X_train_1d('+str(type(X_train_1d))+')'+', '+str(X_train_1d.shape))
    logging.info('MNIST: Y_train_1d('+str(type(Y_train_1d))+')'+', '+str(Y_train_1d.shape))
    logging.info('MNIST: X_test_1d('+str(type(X_test_1d))+')'+', '+str(X_test_1d.shape))
    logging.info('MNIST: Y_test_1d('+str(type(Y_test_1d))+')'+', '+str(Y_test_1d.shape))

    # With the help of GridSearchCV and parameters grid, create a model.
    svcMnist = svm.SVC(probability=True, decision_function_shape='ovo')
    if args.fast is True:
        param_grid={'C':[0.1],'gamma':[0.0001],'kernel':['poly']}
    else:
        param_grid={'C':[0.05, 0.1, 0.2],
                    'gamma':[0.00005, 0.0001, 0.0002],
                    'kernel':['poly']}
    svc = svm.SVC(probability=True, decision_function_shape='ovo')
    svmnist = GridSearchCV(svcMnist, param_grid,
                           cv=4, n_jobs=6, verbose=3)

    svmnist.fit(X_train_1d, Y_train_1d.ravel())
    Y_pred = svmnist.predict(X_test_1d)
    logging.info('MNIST: Y_pred('+str(type(Y_pred))+')'+', '+str(Y_pred.shape))
    logging.info('MNIST: Y_test_1d('+str(type(Y_test_1d))+')'+', '+str(Y_test_1d.shape))
    logging.info('MNIST Accuracy('+str(accuracy_score(Y_pred,Y_test_1d)*100)+'%)')
    logging.info('Used parameters: '+str(svmnist.best_params_))
    logging.info('Used estimator : '+str(svmnist.best_estimator_))
    logging.info('Classes        : '+str(svmnist.classes_))
    logging.info('support_vectors: '+str(svmnist.best_estimator_.support_vectors_.shape))
    time.sleep(2)

    logging.info('\n'+str(classification_report(Y_test_1d, Y_pred)))
    cmMnist = confusion_matrix(Y_test_1d, Y_pred)
    #logging.info('confusion matrix: '+str(cmMnist))
    jplot.plot_confusion_matrix('mnist', cmMnist, svmnist.classes_, DIR_OUTPUT)

    jplot.svm_decision_multi('mnist', svmnist,
                             X_train_1d, Y_train_1d,
                             X_test_1d,  Y_test_1d,
                             DIR_OUTPUT,
                             lxRange=[-10, 10])

def main():

    DIR_OUTPUT = os.path.join(HERE_RELATIVE, '_output')

    parser = argparse.ArgumentParser(description=textwrap.dedent('''Classification by SVM.'''),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--fast', action='store_true', default=False,
        help=textwrap.dedent('''Fast run for development purposes. Uses e.g. a limited dataset.'''))
    parser.add_argument('--mnist', action='store_true', default=False,
        help=textwrap.dedent('''Runs additional processing on the MNIST dataset.'''))
    args = parser.parse_args()
    logging.info(args)

    timeStart = datetime.now()

    jhelper.recreateDir(DIR_OUTPUT)

    process_bricks(args, DIR_OUTPUT)
    if args.mnist is True:
        process_mnist(args, DIR_OUTPUT)

    timeEnd = datetime.now()
    timeDelta = timeEnd - timeStart
    logging.info('Elapsed time: '+str(timeDelta))

if __name__ == "__main__":
    main()
