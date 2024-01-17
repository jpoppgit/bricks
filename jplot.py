#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Plot lib.
"""
import os
import logging

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

import matplotlib.pyplot as plt

import jhelper

# https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Histogram.html
LCOLORS = []
LCOLORS.append('blue')
LCOLORS.append('firebrick')
LCOLORS.append('green')
LCOLORS.append('darkorange')
LCOLORS.append('aquamarine')
LCOLORS.append('darkgreen')
LCOLORS.append('tomato')
LCOLORS.append('mediumpurple')
LCOLORS.append('yellowgreen')
LCOLORS.append('darkslateblue')
LCOLORS.append('saddlebrown')
LCOLORS.append('indigo')
LCOLORS.append('lawngreen')
LCOLORS.append('mintcream')
LCOLORS.append('plum')

def get_colors(nColors):
    lsColors = []
    for n in range(0, nColors):
        lsColors.append(LCOLORS[n])
    return lsColors

def plot_class_distribution(sTag, DIR_OUTPUT, l_base_categories, y_train, y_test):
    logging.info('#y_train('+str(len(y_train))+')'+' #y_test('+str(len(y_test))+')')

    layout = go.Layout(
        title=sTag+': class/label distribution',
        xaxis=dict(title="Class"),
        yaxis=dict(title="Occurences"),
        barmode='group'
    )

    y_train_classes = list(set(y_train))
    y_test_classes  = list(set(y_test))
    y_train_classes.sort()
    y_test_classes.sort()
    logging.info('y_train_classes('+str(y_train_classes)+')'+' y_test_classes('+str(y_test_classes)+')')
    if y_train_classes == y_test_classes:
        logging.info('List of classes are equal ... OK.')
    else:
        logging.error('List of classes are NOT equal!')
        sys.exit()

    # - determine occurances
    n_class_occurances_train = []
    for current_class in y_train_classes:
        o = y_train.count(current_class)
        n_class_occurances_train.append(o)

    n_class_occurances_test = []
    for current_class in y_test_classes:
        o = y_test.count(current_class)
        n_class_occurances_test.append(o)

    # - generate x-labeling
    s_y_train_classes = []
    for i, current_class in enumerate(y_train_classes):
        s_y_train_classes.append(str(l_base_categories[i])+' / '+str(current_class))
    s_y_test_classes = []
    for i, current_class in enumerate(y_test_classes):
        s_y_test_classes.append(str(l_base_categories[i])+' / '+str(current_class))

    fig = go.Figure(data=[
            go.Bar(name='train', x=s_y_train_classes, y=n_class_occurances_train),
            go.Bar(name='test',  x=s_y_test_classes,  y=n_class_occurances_test)
            ],
            layout=layout
        )
    fig.show()
    pio.write_image(fig, os.path.join(DIR_OUTPUT, sTag+'_class_distribution.png'),
        scale=1, width=1280, height=720)


def plot_df(df_in):
    #logging.info(df_in.keys())
    logging.info(df_in.shape)
       
    l_targets_int = df_in['target_int'].unique().tolist()
    logging.info('l_targets_int: '+str(l_targets_int))
    logging.info('\n'+str(df_in.head()))
    
    l_augmentations = df_in['augmentation'].unique().tolist()
    logging.info('augmentations: '+str(l_augmentations))
    
    nItems = len(l_targets_int) * len(l_augmentations)
    [n_int, n_squared] = jhelper.nextPerfectSquare(nItems)
    nRows = n_int
    nCols = n_int
    logging.info('subplot layout: nItems('+str(nItems)+')'+' nRows('+str(nRows)+')'+' nCols('+str(nCols)+')')

    for i_target, i_target_int in enumerate(l_targets_int):

        # get all rows for current target
        df_current_target = df_in[df_in['target_int'] == i_target_int]
        n_samples_total = df_current_target.shape[0]

        # -- limit plotted samples
        if n_samples_total > 50:
            df_current_target = df_current_target.iloc[np.r_[0:25, -25:0]]
            n_samples = df_current_target.shape[0]

            sLog  = 'target_int('+str(i_target_int)+')'+' '+'n_samples_total('+str(n_samples_total)+')'
            sLog += ' --> limiting to n_samples('+str(n_samples)+')'+' for plotting'
            logging.info(sLog)
        else:
            n_samples = n_samples_total

        [n_int, n_squared] = jhelper.nextPerfectSquare(n_samples)
        nRows = n_int
        nCols = n_int
        sLog  = 'target_int('+str(i_target_int)+')'+' '+'n_samples('+str(n_samples)+')'
        sLog += ' nRows('+str(nRows)+')'+' nCols('+str(nCols)+')'
        logging.info(sLog)

        lsTitleSubplots = []
        for i in range(0, n_samples):
            lsTitleSubplots.append('title '+str(i))

        fig = make_subplots(rows=nRows,
                            cols=nCols,
                            subplot_titles=tuple(lsTitleSubplots)
                            )

        # iterate through each row of current dataframe
        rowCnt = 1
        colCnt = 1
        dfRowCnt = 0
        for _, df_row in df_current_target.iterrows():
            #logging.info('r/c('+str(rowCnt)+'/'+str(colCnt)+')'+', '+str(df_row["augmentation"])+', '+str(df_row["target_int"]))

            fig.add_trace(px.imshow(df_row['image']).data[0],
                          row=rowCnt,
                          col=colCnt)

            lsTitleSubplots[dfRowCnt] = str(df_row["augmentation"])

            # row/col counting
            colCnt += 1
            if colCnt > nCols:
                rowCnt += 1
                colCnt = 1

            dfRowCnt += 1

        sTitle  = 'Target('+str(i_target_int)+')'
        sTitle += ' #samples('+str(n_samples)+'/'+str(n_samples_total)+')'

        for i, sTitleSubplots in enumerate(lsTitleSubplots):
            fig.layout.annotations[i].update(text=sTitleSubplots, font_size=8)
        fig.update_layout(title_text=sTitle)
        fig.show()


def plot_confusion_matrix(sTag, cm, l_classes, DIR_OUTPUT):
    # Define the layout for the heatmap
    layout = go.Layout(
        title=sTag+': confusion matrix',
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label")
    )

    # Create the heatmap trace
    trace = go.Heatmap(z=cm, 
                x=l_classes,
                y=l_classes,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale="Viridis")

    # Create the figure and plot it
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    pio.write_image(fig, os.path.join(DIR_OUTPUT, sTag+'_confusion_matrix.png'),
        scale=1, width=1280, height=720)


def getClassPairs(model):
    # Class pairs from list of classes
    l_classes_pairs = []
    for idxa, a in enumerate(model.classes_):
        for idxb, b in enumerate(model.classes_[idxa+1:]):
            sLog  = '#'+str(idxa)+'/'+str(idxb)
            sLog += ' ('+str(a)+'/'+str(b)+')'
            #logging.info(sLog)
            l_classes_pairs.append([a,b])
    logging.info('l_classes_pairs: '+str(l_classes_pairs))
    return l_classes_pairs


def svm_decision(fig, iCol, lxRange, model,
                 sTag, x_samples, y_samples):

    y_pred_decision = model.decision_function(x_samples)
    logging.info('The following array is a decision function of the samples for each class in the model.')
    logging.info('Returns pairwise comparison of classes. e.g. [AB, AC, AD, BC, BD, CD].')
    logging.info('y_pred_decision: #'+str(len(y_pred_decision))+' '+str(y_pred_decision))

    # get number of bins
    nbinsx = ( abs(lxRange[0]) + abs(lxRange[1]) ) * 2

    # Class pairs from list of classes
    l_classes_pairs = getClassPairs(model)
    nRows = len(l_classes_pairs)

    # generate N colours
    lsColors = get_colors(len(model.classes_))

    # go over each class-pair
    for idxRow in range(0, nRows):
        iRow = idxRow+1

        lidxClassPair = l_classes_pairs[idxRow]
        # print('lidxClassPair: '+str(lidxClassPair))

        lScore = []
        lScoreIdx0 = np.array([])
        lScoreIdx1 = np.array([])
        # go over each sample
        for idxPred, lPredDecision in enumerate(y_pred_decision):
            score = lPredDecision[idxRow]
            lScore.append(score)

            # decide for one label
            if score > 0:
                idxClassPairWinner = 0
                lScoreIdx0 = np.append(lScoreIdx0, np.array([score]))
            else:
                idxClassPairWinner = 1
                lScoreIdx1 = np.append(lScoreIdx1, np.array([score]))

            classWinnerPerHyperplane = lidxClassPair[idxClassPairWinner]

            lPredDecisionFormated = ['{:10.6f}'.format(elem) for elem in lPredDecision]
            sLog = ' '+str(lPredDecisionFormated)
            sLog += ' cpair#('+str(idxRow)+')'
            sLog += ' classes('+str(lidxClassPair)+')'
            sLog += ' '+str("{:8.4f}".format(score))
            sLog += ' -> idx('+str(idxClassPairWinner)+')'
            sLog += ' -> '+str(classWinnerPerHyperplane)
            #logging.info(sLog)

        sColorIdx0 = lsColors[lidxClassPair[0]]
        fig.add_trace(go.Histogram(x=lScoreIdx0,
                      marker=go.histogram.Marker(color=sColorIdx0),
                      name=sTag+', '+str(lidxClassPair)+', class '+str(lidxClassPair[0]),
                      nbinsx=nbinsx
                      ),
                row=iRow, col=iCol
                )

        sColorIdx1 = lsColors[lidxClassPair[1]]
        fig.add_trace(go.Histogram(x=lScoreIdx1,
                      marker=go.histogram.Marker(color=sColorIdx1),
                      name=sTag+', '+str(lidxClassPair)+', class '+str(lidxClassPair[1]),
                      nbinsx=nbinsx
                      ),
                row=iRow, col=iCol
                )

        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.7)

        # fig.add_vline(x=-1, line_width=1,
        #             row=iRow, col=iCol, 
        #             line_dash="dash", line_color="grey")
        fig.add_vline(x=-0, line_width=1,
                    row=iRow, col=iCol,
                    line_dash="dash", line_color="grey")
        # fig.add_vline(x=+1, line_width=1,
        #             row=iRow, col=iCol,
        #             line_dash="dash", line_color="grey")


def svm_decision_multi(sTag,
                       model,
                       x_train, y_train,
                       x_test,  y_test,
                       DIR_OUTPUT,
                       lxRange=[-4, 4]):

    l_classes_pairs = getClassPairs(model)
    nRows = len(l_classes_pairs)

    l_subplot_titles = []
    l_subplot_titles.append('train set, class pair: '+str(l_classes_pairs[0]))
    l_subplot_titles.append('test set, class pair: '+str(l_classes_pairs[0]))
    for class_pair in l_classes_pairs[1:]:
        l_subplot_titles.append('class pair: '+str(class_pair))
        l_subplot_titles.append('class pair: '+str(class_pair))

    fig = make_subplots(rows=nRows, cols=2,
        subplot_titles=l_subplot_titles
        )

    svm_decision(fig, 1, lxRange, model,
                 'train', x_train, y_train)
    svm_decision(fig, 2, lxRange, model,
                 'test', x_test, y_test)

    # ideally this can be detected outside this function
    cntAxis = 1
    for idxClass, class_pair in enumerate(l_classes_pairs):
        if idxClass > 0:
            fig['layout']['xaxis'+str(cntAxis)]['range'] = lxRange; cntAxis+=1
            fig['layout']['xaxis'+str(cntAxis)]['range'] = lxRange; cntAxis+=1
        else:
            fig['layout']['xaxis']['range']  = lxRange; cntAxis+=1
            fig['layout']['xaxis2']['range'] = lxRange; cntAxis+=1

    fig.update_layout(title_text=sTag+': SVM decision boundaries')
    fig.show()
    pio.write_image(fig, os.path.join(DIR_OUTPUT, sTag+'_svm_decision.png'),
        scale=1, width=1280, height=720)
