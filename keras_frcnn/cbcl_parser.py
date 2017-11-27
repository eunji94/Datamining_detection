# -*- coding: utf-8 -*-

"""
Created on Sat Nov 18 12:57:02 2017
@author: Data Mining Project : datamining
"""

import cv2
#from PIL import Image
#from PIL import ImageDraw
import numpy as np
import os
import pandas as pd
import random


def get_data():
    image_path = os.listdir("data/Original")

    filename_list = []

    for i in range(len(image_path)):
        filename_list.append(image_path[i][:-4])

    all_data = []
    trng_data = []
    val_data = []

    class_count = {1:0, 2:0, 3:0, 4:0, 5:0}

    class_mapping = {1: 'car', 2: 'pedestrian', 3: 'bicycle', 5: 'tree', 4: 'building'}

    bicycle = []

    for i in range(len(filename_list)):
        # image
        # img = cv2.imread("./data/Original/" + filename_list[i] + '.jpg')
        file_path = "data/Original/" + filename_list[i] + '.JPG'
        # Annotation
        anno_csv = pd.read_csv("data/Annotations/" + filename_list[i] + '.csv')
        neg = False
        annotation_data = {'filepath': file_path, 'width': 1280, 'height': 960, 'bboxes': [], 'neg': neg}

        if anno_csv.iloc[0,0] == 0:
            neg = True

        if not neg:
            for j in range(anno_csv.shape[0]):

                box = {}
                box['class'] = anno_csv.iloc[j,0]

                # counting class                
                class_count[anno_csv.iloc[j,0]] += 1

                if anno_csv.iloc[j,0] == 3:
                    tmp = int(filename_list[i][-5:])
                    if tmp not in bicycle:

                        bicycle.append(tmp)
                
                box['x1'] = anno_csv.iloc[j,1]
                box['x2'] = anno_csv.iloc[j,2]
                box['y1'] = anno_csv.iloc[j,3]
                box['y2'] = anno_csv.iloc[j,4]

                annotation_data['bboxes'].append(box)

        all_data.append(annotation_data)
#        print(filename_list[i] + '.jpg' + ' complete')


    # divide training and validating data
    random.shuffle(bicycle)
    prop = 0.7

    number_of_bicycle = len(bicycle)

    train_bicycle = bicycle[:int(prop * number_of_bicycle)]

    val_bicycle = bicycle[int(prop * number_of_bicycle):]

    all_list = [i for i in range(len(all_data))]

    else_list = list(set(all_list)- set(bicycle))

    random.shuffle(else_list)

    num_of_else_list = len(else_list)

    train_list = else_list[:int(prop * num_of_else_list)]

    val_list = else_list[int(prop * num_of_else_list):]

    for i in range(len(all_data)):

        if i in train_list:
            trng_data.append(all_data[i])

        elif i in val_list:
            val_data.append(all_data[i])

        elif i in train_bicycle:
            trng_data.append(all_data[i])

        elif i in val_bicycle:
            val_data.append(all_data[i])

    return trng_data, val_data, class_count, class_mapping
