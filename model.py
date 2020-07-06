
## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## parts are from Joseph Nelson's example at:
## https://colab.research.google.com/drive/19LDyrwycB7-p_0DzXit0hYFxz0kI8KU_

## and also some parts in the run_inference_for_single_image is from Edje Electronics example:
## https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors


import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import sys
from Model.utils import label_map_util
from Model.utils import visualization_utils as vis_util

def handle_call(image_path, output_path):
    """
    image_path = the path to the image directory where the to predicted images should be stored
    output_path = the path to directory where the predicted images should be stored

    This function setsup a detection graph based on the model
    """
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # You can chose between the 10k steps trained model and the 50k steps one. 
    PATH_TO_CKPT = r".\Model\10k.pb"

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = r".\Model\trafficcones_label_map.pbtxt"


    PATH_TO_TEST_IMAGES_DIR = image_path

    # list of all images in passed image directory
    assert os.path.isfile(PATH_TO_CKPT)
    assert os.path.isfile(PATH_TO_LABELS)
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
    assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
    print(TEST_IMAGE_PATHS)

    # setup of the detection graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=3, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # if image directory isn't empty, run prediction on the found images
    if (TEST_IMAGE_PATHS):
        return inference_on_images(TEST_IMAGE_PATHS, output_path, detection_graph, category_index), TEST_IMAGE_PATHS


def run_inference_for_single_image(image, detection_graph):
    """
    image = image file the detection graph predicts
    detection_graph = the tensorflow detection graph itself
    """
    with detection_graph.as_default():
        with tf.Session() as sess:

            # Input tensor is the image
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Output tensors are the detection boxes, scores, and classes
            # Each box represents a part of the image where a particular object was detected
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represents level of confidence for each of the objects.
            # The score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            # Number of objects detected
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image})

    return {"boxes":boxes, "scores":scores, "classes":classes, "num":num}

def inference_on_images(images, output_path, detection_graph, category_index):
    """
    images = list of image paths
    output_path = path to output directory
    detection_graph = tensorflow detection graph
    category_index = category indexes needed for visualization
    """
    #loop over images in image list
    for image_path in images:
        image = cv2.imread(image_path)
        #image = cv2.resize(image, (416, 416), interpolation = cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
  
        output_dict = run_inference_for_single_image(image_expanded, detection_graph)
        #print(output_dict)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(output_dict["boxes"]),
            np.squeeze(output_dict["classes"]).astype(np.int32),
            np.squeeze(output_dict["scores"]),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.6)

        # saving of predicted images
        image_name = image_path.split("\\")[-1]
        cv2.imwrite(os.path.join(output_path , image_name), image)
        
        return output_dict
