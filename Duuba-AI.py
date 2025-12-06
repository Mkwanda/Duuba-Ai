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
from pathlib import Path

# Third-party imports
import streamlit as st

# Lazy imports for heavy packages
try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

# TensorFlow will be imported lazily when needed
tensorflow_available = False
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tensorflow_available = True
except ImportError:
    pass

# Check all dependencies are available
def check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python-headless")
    if Image is None:
        missing.append("Pillow")
    
    if missing:
        st.error(f"""
        ❌ Missing dependencies: {', '.join(missing)}
        
        These packages are being installed. Please refresh this page in a moment.
        If the error persists after 5 minutes, the build environment may have timed out.
        """)
        return False
    return True

# Check dependencies on app startup
if not check_dependencies():
    st.stop()

# Now it's safe to use these imports
import numpy as np
import cv2
from PIL import Image

# TensorFlow will be imported when model is loaded (lazy import)
def ensure_tensorflow():
    """Ensure TensorFlow is available, install if needed."""
    global tensorflow_available, tf
    if not tensorflow_available:
        try:
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            tensorflow_available = True
        except ImportError:
            st.error("TensorFlow is required but not installed. Please ensure TensorFlow CPU is available.")
            st.stop()
    return tf

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

# -----------------------------
# Optional profile links (set these to your URLs to show links in the sidebar)
# -----------------------------
# Example: PROFILE_LINKEDIN = 'https://www.linkedin.com/in/your-profile'
#          PROFILE_TWITTER = 'https://twitter.com/your-handle'
PROFILE_LINKEDIN = 'https://www.linkedin.com/in/mubarak-kwanda/'
PROFILE_TWITTER = 'https://twitter.com/mbrkkwanda'
# Default model URL (Google Drive share link provided by user)
DEFAULT_MODEL_URL = 'https://drive.google.com/file/d/1Biw-K0DlbOAxGVEm4wy2LLGdDSuZl1Wg/view?usp=sharing'
# Ensure workspace folders exist
for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# Labels used by this model
labels = [{'name':'Healthy Pod', 'id':1}, {'name':'Infected Pod', 'id':2}]

# Ensure label map file exists
if not os.path.exists(files['LABELMAP']):
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


# -----------------------------
# Load trained detection model from checkpoint
# -----------------------------
import tensorflow as tf

# Try importing TF Object Detection API
try:
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util as od_config_util
    TFOD_AVAILABLE = True
except ImportError:
    st.error("""
    ⚠️ TensorFlow Object Detection API not installed.
    
    For local development, install with:
    ```bash
    pip install tf-models-official
    ```
    """)
    TFOD_AVAILABLE = False
    st.stop()

# Build the model and restore weights from SavedModel (cached for fast reruns)
@st.cache_resource
def load_model():
    """Load the detection model from SavedModel format (cached)."""
    # Ensure TensorFlow is available
    tf = ensure_tensorflow()
    
    savedmodel_dir = os.path.join(paths['CHECKPOINT_PATH'], 'export', 'saved_model')
    
    # Fallback: if export doesn't exist, try direct checkpoint load
    if not os.path.exists(savedmodel_dir):
        st.warning(f"SavedModel not found at {savedmodel_dir}. Attempting legacy checkpoint load...")
        # For legacy checkpoints, we'd need object_detection API which is too heavy
        # Instead, raise a helpful error
        raise RuntimeError(
            f"SavedModel not found. Please ensure the model has been exported.\n"
            f"Expected location: {savedmodel_dir}"
        )
    
    # Load the SavedModel
    detection_model = tf.saved_model.load(savedmodel_dir)
    return detection_model

def model_exists():
    """Return True if the exported SavedModel exists locally."""
    savedmodel_dir = os.path.join(paths['CHECKPOINT_PATH'], 'export', 'saved_model')
    return os.path.exists(os.path.join(savedmodel_dir, 'saved_model.pb'))

def ensure_model_available():
    """Ensure model files are present; if not, attempt to download using MODEL_URL env var.

    Raises RuntimeError if model is still missing after attempted download.
    """
    if model_exists():
        return

    model_url = os.environ.get('MODEL_URL') or globals().get('DEFAULT_MODEL_URL')
    if not model_url:
        raise RuntimeError('Model not found and MODEL_URL not provided.')

    # Call the downloader script in the repo root
    downloader = Path(__file__).parent / 'download_model.py'
    if not downloader.exists():
        raise RuntimeError('Model missing and downloader not found.')
    
    # Import downloader and run
    import subprocess
    result = subprocess.run(
        ['python', str(downloader), '--url', model_url],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f'Model download failed: {result.stderr}')

# Try to load the model
try:
    ensure_model_available()
    detection_model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.info("Please ensure your model has been exported to SavedModel format.")
    st.stop()
try:
    ensure_model_available()
except Exception as e:
    # When running as a module this will propagate; for Streamlit show a user-friendly message
    try:
        st.error(f'Model unavailable: {e}')
    except Exception:
        pass
    raise


@st.cache_resource
def get_detect_fn():
    """Return a cached detect_fn that uses the loaded model."""
    @tf.function
    def detect_fn(image):
        """Run detection model on a single input tensor and return processed detections."""
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    return detect_fn

detect_fn = get_detect_fn()


# -----------------------------
# Image I/O and helper functions
# -----------------------------
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import io
import warnings
warnings.filterwarnings("ignore")

# Create a mapping from label ids to display names for visualization
# (replaces label_map_util which requires object_detection API)
def create_category_index():
    """Create category index from labels."""
    category_index = {}
    for label in labels:
        category_index[label['id']] = {'id': label['id'], 'name': label['name']}
    return category_index

category_index = create_category_index()


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
    st.warning("⚠️ **Under Development** — This model is in pilot phase and may have errors. Use results for reference only.")
    # Show profile links if configured (set PROFILE_GITHUB and PROFILE_LINKEDIN near top of file)
        # (Profile links moved to footer)

# Sidebar status placeholder for processing alerts
status_placeholder = st.sidebar.empty()

st.title("Duuba-AI - Cocoa 🍍 Monitoring made Easy")
st.info("📌 **Pilot Model** — This detection model is under active development. Predictions may not be 100% accurate. Please validate results independently.")


# live camera view removed (was here). To re-enable, reintroduce the function
# and the UI control. Keeping this comment to make it easy to restore later.



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

    # Notify user and run inference with spinner
    status_placeholder.info('Starting detection...')
    try:
        with st.spinner('Running detection — this may take a few seconds...'):
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
        status_placeholder.success('Detection finished')
    except Exception as e:
        status_placeholder.error(f'Detection failed: {e}')
        raise
    finally:
        # keep status for a short while then clear (non-blocking)
        pass 

    # Post-process detection  utputs
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

    # Decide classification based on detections that meet a confidence threshold.
    # Provide a sidebar slider so the user can tune the threshold to match the visualization.
    conf_thresh = st.sidebar.slider('Confidence threshold', 0.3, 0.95, 0.8, step=0.05)

    # Gather classes and scores (apply label id offset)
    classes = detections['detection_classes'] + label_id_offset
    scores = detections['detection_scores']

    # Find indices of detections above threshold
    valid_idxs = [i for i, s in enumerate(scores) if s >= conf_thresh]

    if not valid_idxs:
        st.write('No detection seen above the confidence threshold. Try lowering the threshold.')
    else:
        # Aggregate scores per class (sum of confidences) to pick the dominant class
        agg = {}
        for i in valid_idxs:
            cls = int(classes[i])
            agg[cls] = agg.get(cls, 0.0) + float(scores[i])

        # Choose class with highest aggregated score
        chosen_cls = max(agg, key=agg.get)
        # Compute a representative score for display (max score among that class)
        chosen_score = max(float(scores[i]) for i in valid_idxs if int(classes[i]) == chosen_cls) * 100

        if chosen_cls == 1:
            st.balloons()
            st.sidebar.success("Classification:  HEALTHY POD")
            healthy_msg(chosen_score, 0)
        elif chosen_cls == 2:
            st.sidebar.warning("Classification: INFECTED POD")
            infected_msg(chosen_score, 0)
        else:
            st.write('No detection seen Please upload new image')

def draw_boxes_on_image(image_np, boxes, classes, scores, category_index, min_score_thresh=0.5):
    """Draw bounding boxes and labels on image (replaces viz_utils.visualize_boxes_and_labels_on_image_array)."""
    im_height, im_width = image_np.shape[:2]
    
    for i in range(len(scores)):
        if scores[i] < min_score_thresh:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        ymin, xmin, ymax, xmax = boxes[i]
        left, right = int(xmin * im_width), int(xmax * im_width)
        top, bottom = int(ymin * im_height), int(ymax * im_height)
        
        class_id = int(classes[i]) + 1  # label_id_offset
        score = scores[i]
        
        # Get class name
        class_name = category_index.get(class_id, {}).get('name', f'Class {class_id}')
        
        # Draw bounding box
        color = (0, 255, 0) if class_id == 1 else (0, 0, 255)  # Green for healthy, red for infected
        cv2.rectangle(image_np, (left, top), (right, bottom), color, 3)
        
        # Draw label
        label = f'{class_name}: {score:.2%}'
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Background for text
        text_x, text_y = left, max(top - 10, 0)
        cv2.rectangle(image_np, (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), color, -1)
        
        # Draw text
        cv2.putText(image_np, label, (text_x + 2, text_y - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return image_np

# Visualize bounding boxes and labels on the image
image_np_with_detections = draw_boxes_on_image(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    min_score_thresh=0.5)

# Enhance visualization: draw thicker boxes and label backgrounds for better visibility
min_score = 0.5
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

# Prepare a presentable 800x800 final image using PIL
try:
    pil_img = Image.fromarray(annotated_rgb)

    # Resize while preserving aspect ratio to fit within 800x800
    target_size = (800, 800)
    pil_copy = pil_img.copy()
    pil_copy.thumbnail(target_size, Image.LANCZOS)

    # Create white background and paste centered
    final_img = Image.new('RGB', target_size, (255, 255, 255))
    paste_x = (target_size[0] - pil_copy.width) // 2
    paste_y = (target_size[1] - pil_copy.height) // 2
    final_img.paste(pil_copy, (paste_x, paste_y))

    # Draw a small legend and title on the final image
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype('arial.ttf', 18)
        font_small = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    title = 'Detections'
    tw, th = draw.textsize(title, font=font)
    draw.text(((target_size[0] - tw) / 2, 8), title, fill=(0, 0, 0), font=font)

    # Legend (top-left corner)
    legend_x = 12
    legend_y = 40
    legend_gap = 6
    legend_items = [(1, 'Healthy Pod', (0, 200, 0)), (2, 'Infected Pod', (200, 0, 0))]
    box_size = 16
    for idx, name, color in legend_items:
        draw.rectangle([legend_x, legend_y, legend_x + box_size, legend_y + box_size], fill=color)
        draw.text((legend_x + box_size + 6, legend_y - 2), name, fill=(0, 0, 0), font=font_small)
        legend_y += box_size + legend_gap

    # Convert final image to bytes for Streamlit display and download
    buf = io.BytesIO()
    final_img.save(buf, format='PNG')
    buf.seek(0)
    img_bytes = buf.getvalue()

    # Show the 800x800 presentable image
    st.image(final_img, caption='Annotated (800x800)', use_column_width=False, width=800)

    # Provide downloads: full-size PNG and open in new tab link
    st.download_button('Download annotated (800x800)', data=img_bytes, file_name='annotated_800.png', mime='image/png')
    data_url = "data:image/png;base64," + __import__('base64').b64encode(img_bytes).decode('utf-8')
    st.markdown(f"[Open annotated image in new tab]({data_url})", unsafe_allow_html=True)

except Exception as e:
    # Fallback to original display if anything goes wrong
    st.image(annotated_rgb, caption='Detections', use_column_width=True)
    st.write('Could not create 800x800 presentable image:', e)

# -----------------------------
# Footer: show profile links if configured
# -----------------------------
try:
    if PROFILE_LINKEDIN or PROFILE_TWITTER:
        # small inline SVG icons (kept simple & lightweight)
        linkedin_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
                        'style="vertical-align:middle;margin-right:6px;fill:#0A66C2"><title>LinkedIn</title>'
                        '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.038-1.852-3.038-1.853 0-2.135 1.446-2.135 2.94v5.667H9.351V9h3.414v1.561h.049c.476-.9 1.637-1.852 3.369-1.852 3.603 0 4.268 2.372 4.268 5.459v6.284zM5.337 7.433a2.062 2.062 0 1 1 0-4.124 2.062 2.062 0 0 1 0 4.124zM6.962 20.452H3.712V9h3.25v11.452z"/></svg>')

        x_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
                 'style="vertical-align:middle;margin-right:6px;fill:#000000"><title>X</title>'
                 '<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24h-6.6l-5.17-6.76-5.91 6.76h-3.308l7.73-8.835L2.42 2.25h6.76l4.69 6.231 5.386-6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>')

        parts = []
        if PROFILE_LINKEDIN:
            parts.append(f'<a href="{PROFILE_LINKEDIN}" target="_blank" rel="noopener" '
                         f'style="text-decoration:none;color:inherit;margin:0 8px">{linkedin_svg}<span style="vertical-align:middle">LinkedIn</span></a>')
        if PROFILE_TWITTER:
            parts.append(f'<a href="{PROFILE_TWITTER}" target="_blank" rel="noopener" '
                         f'style="text-decoration:none;color:inherit;margin:0 8px">{x_svg}<span style="vertical-align:middle">X</span></a>')

        footer_html = ('<div style="text-align:center; margin-top:20px;">' + ' | '.join(parts) +
                       '<br><small style="color:#666">Connect with the project owner</small></div>')
        st.markdown('---')
        st.markdown(footer_html, unsafe_allow_html=True)
except Exception:
    pass
