#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:47:07 2019

@author: chenanyi
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import yaml

class color_threshold():
    def __init__(self,img_name,num_clusters,resize_img,use_hsv_threshold,position_feature,database):
        '''
        Input Parameter: 
            ima_name: The name of image to process.
            num_clusters: the number of cluster groups
            resize_img: Boolean variable, if true, resize the image to shape[300,300,3] to speed up, 
            otherwise use the original image.
            use_hsv_threshold: Boolean variable, if true, use HSV value to cluster, otherwise use RGB.
            position_feature: Boolean variable, if true, add the position feature to analyze.
        Output: Save the cluster result for each cluster.
        '''
        self.img_name = img_name
        self.num_clusters = num_clusters
        self.use_hsv_threshold = use_hsv_threshold
        self.position_feature = position_feature
        self.database = yaml.load(open(database, 'r'))
        self.detected_color_name = {}
        pixel_array,self.og_img = self.load_pixel(resize_img) 
        self.apply_kmeans(pixel_array)
        
        
    def apply_kmeans(self,pixel_array):
        km = KMeans(n_clusters=self.num_clusters,init='k-means++',n_init=10,max_iter=1000)
        km.fit(pixel_array)
        # Get the cluster label for each pixel
        clusters = km.labels_.tolist()
        df = {'cluster': clusters}
        frame = pd.DataFrame(df, index = [clusters] , columns = ['cluster'])
        total_pixel_number = sum(frame['cluster'].value_counts())
        for key,values in frame['cluster'].value_counts().items():
            print('The ratio of cluster '+ str(key) + ' is ' + str(round(values*100/total_pixel_number,1)))
            print()
        print('The number of pixel in each cluster: ')
        print(frame['cluster'].value_counts())
        print()
        for i in range(self.num_clusters):
            # If we set use_hsv_threshold is True, the result is HSV value, otherwise is RGB value.
            print('The pixel value of cluster center '+ str(i) + ' is')
            cluster_center_hsv_value = km.cluster_centers_[i].astype(np.int16)
            print(cluster_center_hsv_value)
            names = self.add_name(cluster_center_hsv_value)
            self.detected_color_name[i] = '_'.join(names)
            print('The tags are ',names)
            print()
        self.save_cluster_result(clusters)
        
    def load_pixel(self,resize_img):
        # Get the pixel value in the image to do kmeans.
        img = cv2.imread(self.img_name)
        if resize_img:
            img = cv2.resize(img, (300, 300,3), interpolation=cv2.INTER_CUBIC)
        og_img = img.copy()
        if self.use_hsv_threshold:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,w,c = img.shape
        count = 0
        # Add the position feature for k-means method
        if self.position_feature:
            pixel_array = np.empty([h*w,c+2],dtype=np.int16)
            for i in range(h):
                for j in range(w):
                    pixel_array[count,:c] = img[i,j,:]
                    pixel_array[count,c:] = [int(i*255/h),int(j*255/w)]
                    count += 1
        else:
            pixel_array = np.empty([h*w,c],dtype=np.int16)
            for i in range(h):
                for j in range(w):
                    pixel_array[count,:c] = img[i,j,:]
                    count += 1
        return pixel_array,og_img

    def add_name(self,cluster_center_arr):
        name_tag = []
        color_list = ['red','orange','yellow','green',
                      'spring_green','blue','violet'
                     ,'rose','black','white','grey']
        for color in color_list:
            lower,upper = self.database[color]['lower'],self.database[color]['upper']
            if lower[0] <= cluster_center_arr[0] <= upper[0] and lower[1] <= cluster_center_arr[1] <= upper[1] and lower[2] <= cluster_center_arr[2] <= upper[2]:
                name_tag.append(color)
        return name_tag

    
    def save_cluster_result(self,clusters):
        h,w,c = self.og_img.shape
        result = np.zeros([self.num_clusters,h,w,c]) + 255
        count = 0
        for i in range(h):
            for j in range(w):
                cluster_class= clusters[count]
                result[cluster_class,i,j,:] = self.og_img[i,j,:]
                count += 1
        title = 'position_feature' if self.position_feature else 'result_'
        for i in range(self.num_clusters):
            cv2.imwrite(title + self.detected_color_name[i] +str(i) +'.png', result[i,:,:,:])
        print('done!')      
        
x = color_threshold(img_name = 'test1.png',num_clusters = 5,resize_img = False,use_hsv_threshold=True,position_feature = False,database = 'color_hsv.yaml')
#x = color_threshold(img_name = 'color_wheel.png',num_clusters = 11,resize_img = False,use_hsv_threshold=True,position_feature = False,database = 'color_hsv.yaml')
