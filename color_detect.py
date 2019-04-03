#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:50:33 2019

@author: chenanyi
"""

import cv2
import numpy as np
import os
import yaml

class color_detection():
    def __init__(self,img_name,save_mask,threshold,database):
        '''
        Input Parameter: 
            ima_name: The name of image to process
            save_mask: Boolean variable, if true, save the mask array for image
            threshold: Between (0,1). Ignore colors whose area is less than the threshold.
        Output: Save the colors (the total colors are 9 kinds) that appear in the image as images
        '''
        
        self.img_name = img_name
        self.database = yaml.load(open(database, 'r'))
        
        assert os.path.isfile(self.img_name),'The image file is not valid!'
        assert os.path.isfile(database),'The dataset file is not valid!'
        
        img = cv2.imread(self.img_name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        color_list = ['red','orange','yellow','green',
                      'spring_green','blue','violet'
                     ,'rose','black','white','grey']

        for color in color_list:
            # Threshold the HSV image to get the specific colors
            lower,upper = np.array(self.database[color]['lower']),np.array(self.database[color]['upper'])
            # If the HSV value of one pixel in the range[lower,upper],
            # the corresponding pixel value in mask array will be 0, otherwise be 255
            mask = cv2.inRange(hsv, lower, upper)
            # Ignore colors whose area is less than the threshold
            if self.color_area_ratio(mask) >= threshold:
                # Bitwise-AND mask and original image
                res = cv2.bitwise_and(img,img, mask= mask)
                if save_mask:
                    cv2.imwrite(os.path.splitext(img_name)[0]+ '_' + color +'_' + 'mask.png', mask)
                cv2.imwrite(os.path.splitext(img_name)[0] + '_'+ color +'_' + 'result.png', res)
            else:
                print('The area ration of %s color is too small, we will ignore it.' %color)
        print('Finish all color detection')
        
    def color_area_ratio(self,array):
        h,w = array.shape
        count = 0
        for i in range(h):
            for j in range(w):
                if array[i][j] != 0:
                    count += 1
        return count/(h*w)

#color_det = color_detection(img_name = 'colormap.jpeg',save_mask = False,threshold= 0.0,database = 'color_hsv.yaml')
color_det = color_detection(img_name = 'test1.png',save_mask = False,threshold= 0.0,database = 'color_hsv.yaml')
#color_det = color_detection(img_name = 'color_wheel.png',save_mask = False,threshold= 0.0,database = 'color_hsv.yaml')    
