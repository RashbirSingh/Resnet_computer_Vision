#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 04:54:19 2018

@author: apple
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images.png"), output_image_path=os.path.join(execution_path , "imagess.png"))

for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )