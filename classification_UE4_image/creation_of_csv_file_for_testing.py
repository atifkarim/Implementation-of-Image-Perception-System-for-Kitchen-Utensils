#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:57:08 2019

@author: atif
"""
import os

image_path = "/home/atif/machine_learning_stuff/ml_image/test_image_same_number"
csv_file_path = "/home/atif/machine_learning_stuff/ml_image/test_file_same_number.csv"

for a,b,files in os.walk(image_path):
    for file_name in files:
        print(file_name)
        f = open(csv_file_path, 'a')
        f.write(str(file_name))
        f.write('\n')
f.close()