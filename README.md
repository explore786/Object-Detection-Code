# Object-Detection-Code  Live Object Recognition using webcam

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format


# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('C:/Users/My Dell/Anaconda3/Lib/site-packages/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

# label_id_offset = 1

# Number of classes to detect
NUM_CLASSES = 90

# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

def load_labelmap(path):
 
 with tf.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)
    return label_map


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
    for item in label_map.item:  #####
        categories.append({
          'id': item.id,
          'name': item.display_name
      }) 

    return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info(
                'Ignore item %d since it falls outside of requested '
                'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories

def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


label_map = load_labelmap(PATH_TO_LABELS)
categories = convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True)    

#             # Display output
#             ret, frame = cap.read()
#             cv2.imshow('object detection',frame)
            cv2.imshow('object detection', cv2.resize(image_np, (1200,700)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
                
