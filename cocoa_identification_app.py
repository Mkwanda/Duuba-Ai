import streamlit as st
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from keras.models import load_model
from PIL import Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline


paths = {
    
    
    'ANNOTATION_PATH': './annotations',
    'CHECKPOINT_PATH': './my_ssd_mobnetpod'
    
 }


files = {
    'PIPELINE_CONFIG': './my_ssd_mobnetpod/pipeline.config',
    
    'LABELMAP': './annotations/label_map.pbtxt'
}

# Load pipeline config and build a detection model
loaded_model = tf.saved_model.load('my_ssd_mobnetpod\export\saved_model')
#print(list(loaded_model.signatures.keys()))
#configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
#detection_model = model_builder.build(model_config=configs['model'], is_training=False)



# Restore checkpoint
#ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
#ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()
"""
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
"""

#category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

def run():
    st.title("Cocoa 🍍 Classification")
    #print(list(loaded_model.signatures.keys()))
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img_raw = Image.open(img_file).resize((250, 250))
        #st.image(img_raw, use_column_width=False)
        #save_image_path = './upload_images/' + img_file.name
        #with open(save_image_path, "wb") as f:
         #   f.write(img_file.getbuffer())
"""
        IMAGE_PATH = (img_file.name)

        img_uploaded = cv2.imread(IMAGE_PATH)
        image_np = np.array(img_uploaded)
        

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

#print(image_np_with_detections)


        viz_utils.visualize_boxes_and_labels_on_image_array(
                    
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.6,
                    agnostic_mode=False
                    )
        plt.figure()
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        print(plt.show()) 
"""

run()