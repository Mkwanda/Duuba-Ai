#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Duuba-AI Cocoa detection Streamlit app
# This script loads a trained TensorFlow Object Detection model, accepts an uploaded
# image via Streamlit, runs inference to detect cocoa pods (healthy vs infected),
# and visualizes results.

# Standard library imports
from distutils.sysconfig import get_python_inc
import os

# Third-party imports
import streamlit as st

# -----------------------------
# Configuration / constants
# -----------------------------
# Model and file names used by the app
CUSTOM_MODEL_NAME = 'my_ssd_mobnetpod'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# File system layout used throughout the project. Change these if your workspace moves.
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
    'TFJS_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('Cocoa','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('Cocoa','Tensorflow','protoc')
}

# Helpful file paths derived from `paths`
files = {
    'PIPELINE_CONFIG': os.path.join('Cocoa','Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Ensure workspace folders exist (this block can be skipped if folders already present)
for path in paths.values():
    if not os.path.exists(path):
        # These lines were originally written for notebook environments; keep as-is
        if os.name == 'posix':
            get_ipython().system('mkdir -p {path}')
        if os.name == 'nt':
            get_python_inc().system('mkdir {path}')


# Optional setup steps commented out (git clone of TF models, protobuf install, etc.)
# They are useful when preparing a new environment from scratch.
# if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
#     get_ipython().system("git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")
# get_ipython().system('pip install protobuf==3.19.6')


# -----------------------------
# Object detection imports and label map creation
# -----------------------------
import object_detection

# Labels used by this model; update if label ids or names change.
labels = [{'name':'Healthy Pod', 'id':1}, {'name':'Infected Pod', 'id':2}]

# Write the label map file used by the TF OD API
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# TFRecord generation steps are left commented (this project includes training utilities)
# if not os.path.exists(files['TF_RECORD_SCRIPT']):
#     get_ipython().system("git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")


# -----------------------------
# Update pipeline config for transfer learning
# -----------------------------
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Read the pipeline config and merge into protobuf object
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Update a few fields for this fine-tuning setup (num classes, checkpoint, input paths)
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-3')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

# Persist the edited pipeline config back to disk
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)


# -----------------------------
# Load trained detection model from checkpoint
# -----------------------------
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util as od_config_util

# Build the model and restore weights from checkpoint
configs = od_config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()


@tf.function
def detect_fn(image):
    """Run detection model on a single input tensor and return processed detections."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# -----------------------------
# Image I/O and helper functions
# -----------------------------
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Create a mapping from label ids to display names for visualization
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


def format_score(score, digits=2):
    """Safely convert various numeric types to float and round to `digits`.

    Returns a float (rounded) or 0.0 on failure.
    """
    try:
        return round(float(score), digits)
    except Exception:
        try:
            return float(score)
        except Exception:
            return 0.0


def healthy_msg(score1, score2):
    """Display messages/sidebars when a pod is classified as healthy.

    score1, score2: confidence scores (percent) for the top detections.
    """
    s1 = format_score(score1, 2)
    s2 = format_score(score2, 2)
    if s1 > 50.00:
        st.write(f' {s1}% HEALTHY')
        st.sidebar.write("Accuracy Metric: Above 50%")
        st.sidebar.success(f'Accuracy: {s1}% HEALTHY')
    else:
        # No action for low-confidence healthy detection
        pass


def infected_msg(score1, score2):
    """Display messages/sidebars when a pod is classified as infected."""
    s1 = format_score(score1, 2)
    s2 = format_score(score2, 2)
    if s1 > 50.00:
        st.write(f'Accuracy:  {s1}% INFECTED')
        st.sidebar.write("Accuracy Metric: Above 50%")
        st.sidebar.error(f'Accuracy:  {s1}% INFECTED')
    else:
        pass


# -----------------------------
# Streamlit app layout
# -----------------------------
with st.sidebar:
    st.title("Duuba-AI - Cocoa 🍍 Disease Detector")
    st.subheader("Accurate detection of Infected or Healthy Cocoa pods.")
    st.write("Classification: 1 = HEALTHY / 2 = INFECTED")

st.title("Duuba-AI - Cocoa 🍍 Monitoring made Easy")


# -----------------------------
# Image upload flow
# -----------------------------
IMAGE_PATH = st.file_uploader("Choose an Image", type=["jpg", "png"])
if IMAGE_PATH is not None:
    # Read and display the uploaded image
    img_raw = Image.open(IMAGE_PATH).resize((400, 400))
    st.image(img_raw, use_column_width=False)

    # Read uploaded file bytes directly and decode with OpenCV (avoids filesystem writes)
    file_bytes = np.asarray(bytearray(IMAGE_PATH.getbuffer()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error('Failed to decode uploaded image')
        raise RuntimeError('cv2.imdecode returned None')
    image_np = img

    # Convert to tensor and run inference
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # Post-process detection outputs
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Extract top detection classes and scores (convert to percent)
    det_num = detections['detection_classes'] + label_id_offset
    det_score = detections['detection_scores']
    det_score1 = det_score[0] * 100
    det_score2 = det_score[1] * 100

    # Simple heuristic to choose between healthy/infected messages based on top detections
    # NOTE: this logic can be improved to handle multiple detections and thresholds.
    if det_num[0] and det_num[1] == 1:
        st.balloons()
        st.sidebar.success("Classification:  HEALTHY POD")
        healthy_msg(det_score1, det_score2)
    elif det_num[0] and det_num[1] == 2:
        st.sidebar.warning("Classification: INFECTED POD")
        infected_msg(det_score1, det_score2)
    else:
        st.write('No detection seen Please upload new image')

    # Visualize bounding boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    # Enhance visualization: draw thicker boxes and label backgrounds for better visibility
    min_score = 0.8  # same threshold used in viz_utils call above
    h, w, _ = image_np_with_detections.shape
    for i in range(min(5, int(detections.get('num_detections', 0)))):
        score = float(detections['detection_scores'][i])
        if score < min_score:
            continue
        box = detections['detection_boxes'][i]
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
        cls = int(detections['detection_classes'][i]) + label_id_offset
        class_name = category_index.get(cls, {'name': 'N/A'})['name']
        label = f"{class_name}: {int(score * 100)}%"
        # Pick color based on class (green for healthy, red for infected)
        color = (0, 200, 0) if class_name.lower().startswith('healthy') or cls == 1 else (0, 0, 200)
        # Draw thick rectangle
        cv2.rectangle(image_np_with_detections, (left, top), (right, bottom), color, thickness=4)
        # Draw filled rectangle for label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image_np_with_detections, (left, max(0, top - text_h - 10)), (left + text_w, top), color, -1)
        cv2.putText(image_np_with_detections, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Convert to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

    # Main display (fits column)
    st.image(annotated_rgb, caption='Detections', use_column_width=True)

    # Expander for a larger view that 'pops out'
    with st.expander('View large annotated image'):
        # Show a bigger image but cap width for very large images
        display_width = min(1200, w * 2)
        st.image(annotated_rgb, use_column_width=False, width=display_width)

    # Provide a download button so user can save the annotated result
    try:
        ok, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        if ok:
            img_bytes = buffer.tobytes()
            st.download_button('Download annotated image', data=img_bytes, file_name='annotated.png', mime='image/png')
    except Exception:
        # If encoding fails, skip download button silently
        pass
