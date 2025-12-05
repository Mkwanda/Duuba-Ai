#!/usr/bin/env python
# coding: utf-8

from distutils.sysconfig import get_python_inc
import os
import streamlit as st
                                                

# In[2]:

CUSTOM_MODEL_NAME = 'my_ssd_mobnetpod' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# In[3]:

paths = {
	'WORKSPACE_PATH': os.path.join('Cocoa','Tensorflow', 'workspace'),
	'SCRIPTS_PATH': os.path.join('Cocoa','Tensorflow','scripts'),
	'APIMODEL_PATH': os.path.join('Cocoa','Tensorflow','models'),
	'ANNOTATION_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','annotations'),
	'IMAGE_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','images'),
	'MODEL_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','models'),
	'PRETRAINED_MODEL_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','pre-trained-models'),
	'CHECKPOINT_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
	'OUTPUT_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
	'TFJS_PATH':os.path.join('Cocoa','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
	'TFLITE_PATH':os.path.join('Cocoa','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
	'PROTOC_PATH':os.path.join('Cocoa','Tensorflow','protoc')
 }

# ...existing code...
