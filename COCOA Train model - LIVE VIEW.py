#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


files = {
    'PIPELINE_CONFIG':os.path.join('Cocoa','Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# In[5]:


for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            get_ipython().system('mkdir -p {path}')
        if os.name == 'nt':
            get_python_inc().system('mkdir {path}')


# In[6]:




# In[7]:


#if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
 #   get_ipython().system("git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")



# In[15]:


#get_ipython().system('pip install protobuf==3.19.6')


# In[9]:


#VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
#get_ipython().system('python {VERIFICATION_SCRIPT}')


# In[10]:


import object_detection


# In[11]:


#Create Label Map
labels = [{'name':'Healthy Pod', 'id':1}, {'name':'Infected Pod', 'id':2}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


# In[12]:


# Create TF records
#if not os.path.exists(files['TF_RECORD_SCRIPT']):
 #  get_ipython().system("git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")


# In[13]:


#get_ipython().system("python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")
#get_ipython().system("python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")



# In[15]:


# 5. Update Config For Transfer Learning
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# In[16]:


config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])




# In[18]:


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  


# In[19]:


pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-3')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]


# In[20]:


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text) 



# In[26]:


# 8. Load Train Model From Checkpoint

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


# In[27]:


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# In[28]:


# Detect from an Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tkinter
import warnings
warnings.filterwarnings("ignore")
#get_python_version().run_line_magic('matplotlib', 'inline')


# In[31]:


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


# In[32]:
def format_score(score, digits=2):
    try:
        return round(float(score), digits)
    except Exception:
        try:
            return float(score)
        except Exception:
            return 0.0


def healthy_msg(score1, score2):
    s1 = format_score(score1, 2)
    s2 = format_score(score2, 2)
    if s1 > 50.00:
        st.write(f' {s1}% HEALTHY')
        st.sidebar.write("Accuracy Metric: Above 50%")
        st.sidebar.success(f'Accuracy: {s1}% HEALTHY')


def infected_msg(score1, score2):
    s1 = format_score(score1, 2)
    s2 = format_score(score2, 2)
    if s1 > 50.00:
        st.write(f'Accuracy:  {s1}% INFECTED')
        st.sidebar.write("Accuracy Metric: Above 50%")
        st.sidebar.error(f'Accuracy:  {s1}% INFECTED')
    else:
        pass
    #if score2 > 50.00:
     #   st.write(f'Cocoa is {score2}% INFECTED')
    #else:
    #    pass


with st.sidebar:
        
        st.title("Classification of Cocoa pods🍍")
        st.subheader("Accurate detection of Infected or Healthy Cocoa pods. ")
        st.write("Classification: 1 = HEALTHY / 2 = INFECTED")
#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'avacado3.jpg')
st.title("Cocoa 🍍 Classification")
live_video_view()


# In[33]:

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#IMAGE_PATH = st.file_uploader("Choose an Image", type=["jpg", "png"])
while cap.isOpened(): 
    ret, frame = cap.read()
    if frame is not None:
        #img_raw = Image.open(IMAGE_PATH).resize((400, 400))
        #st.image(img_raw, use_column_width=False)
       # save_image_path = './upload_images/' + IMAGE_PATH.name
       # with open(save_image_path, "wb") as f:
        #   f.write(IMAGE_PATH.getbuffer()) 

        #img = cv2.imread(save_image_path)
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()


         #detection msg
        det_num = detections['detection_classes']+label_id_offset
        det_score = detections['detection_scores']
        det_score1 = det_score[0]*100
        det_score2 = det_score[1]*100
        #st.write(detections)
        

#chec 4 detections ( 1 = heathy , 2 = infected)

        if det_num[0] and det_num[1] == 1:
            st.balloons()
            
            st.sidebar.success("Classification:  HEALTHY POD")
            healthy_msg(det_score1,det_score2)

            
        elif det_num[0] and det_num[1] == 2:
            st.sidebar.warning("Classification: INFECTED POD")
            infected_msg(det_score1,det_score2)
        else:
            st.write('No detection seen Please upload new image')  


        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
        plt.switch_backend('TkAgg')
        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
       
        #plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        resut = cv2.show()
           
        st.write(resut)


