######## Webcam Object Detection Using Tensorflow-trained Classifier #########



# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib
from matplotlib import pyplot as plt
import pytesseract
import imutils
import argparse
import time


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import crop_morphology as c_m
import correct_skew as c_s
import readtext as readtext
fst = False
totalpass=0






# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

## Load the label map.

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    #with tf.Session(graph=detection_graph):


# Define input and output tensors (i.e. data) for the object detection classifier

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

# Initialize webcam feed
video = cv2.VideoCapture(0+cv2.CAP_DSHOW)
ret = video.set(3,1280)
ret = video.set(4,720)


checkingSerial = False
it=0
while(True):
##    if(checkingSerial==False):
    ret, frame = video.read()

    frame_expanded = np.expand_dims(frame, axis=0)

##    if(checkingSerial==False):
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    boxesFiltered=[]
    if not fst:
        img2 = np.zeros((512,512,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2,str(totalpass),(10,80), font, 2,(0,153,0),3,cv2.LINE_AA)
        fst=True
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.90)
    
    #if(vis_util.current_obj_isSerial==True):
    #print(vis_util.current_obj_isSerial)
                #[x,y,w,h] = cv2.boundingRect(box)
                #roi = frame_expanded[y:y+h, x:x+w]
    #cv2.imwrite("roi.jpg", box)
        #serialimg = cv2.imencode(".jpg", frame_expanded))
    #print(boxes)
    #image, contours, hierarchy = cv2.findContours(frame_expanded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # All the results have been drawn on the frame, so it's time to display it.
    #roi="roi.jpg"
    cv2.imshow('Object detector', frame)
    

    # Perform the actual detection by running the model with the image as input


    






    final_score = np.squeeze(scores)
    count = 0
    for i in range(100):
        if scores is None or final_score[i] > 0.5:
            count = count + 1
    print('count',count)
    printcount =0;

    for i in classes[0]:
        print(boxes[0])
        printcount = printcount +1
        if(category_index[i]['name']=="sign_readable"):
          #checkingSerial=True
          try:
              coordinates = vis_util.return_coordinates(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=False,
                        line_thickness=8,
                        min_score_thresh=0.90)
          except:
              print("Failed to get coords")
          print(category_index[i]['name'])
          print(coordinates[0])
          print(i)
          it += 1
          if(it>10):
              it=0
          try:
              x= int(coordinates[2])
              y= int(coordinates[0])
              w= int(coordinates[3])
              h= int(coordinates[1])
              
          except:
              
              print("no coordinates")
              ymin = int((boxes[0][0][0]*height))
              xmin = int((boxes[0][0][1]*width))
              ymax = int((boxes[0][0][2]*height))
              xmax = int((boxes[0][0][3]*width))

              Result = np.array(img_np[ymin:ymax,xmin:xmax])
              pass

          roi = frame[y:y+h, x:x+w]

        #im2, contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #x, y, w, h = cv2.boundingRect(contours[i])
        #gaussian_3 = cv2.GaussianBlur(roi, (9,9), 10.0)
        #unsharp_image = cv2.addWeighted(roi, 1.5, gaussian_3, -0.5, 0, roi)

          roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
          cv2.fastNlMeansDenoisingColored(roi,None,15,15,7,21)
          roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
          kernel = np.zeros((3, 3), np.uint8)
        #roi = cv2.dilate(roi, kernel, iterations=1)
          roi = cv2.erode(roi, kernel, iterations=3)
        #roi = cv2.bilateralFilter(roi,9,75,75)
          roi = cv2.medianBlur(roi, 3)
        #edges = cv2.Canny(roi,100,200)
        #img_dilation = c_m.dilate(edges,N=3,iterations=2)
        #kernel = np.ones((5,5), np.uint8)
        #img_dilation = cv2.dilate(roi, kernel, iterations=2)
          roi = cv2.adaptiveThreshold(roi, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
        #ret,roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)

        #plt.subplot(121),plt.imshow(roi,cmap = 'gray')
        #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
          roi = cv2.medianBlur(roi, 3)
        #roi = cv2.bilateralFilter(roi,9,75,75)

          #cv2.imwrite(str(it)+"roi.JPG", roi)
        #contours=c_m.find_components(edges)
        #c_m.process_image(str(it)+"roi.jpg",str(it)+"roi.jpg")
        #API.SetVariable("classify_enable_learning","0");
        #API.SetVariable("classify_enable_adaptive_matcher","0")
        #API.
          r=0
          text = readtext.readtext(roi,r)


          print("{}\n".format(text))
          if(text==None):
            text=str('')
          else:
            img2 = np.zeros((512,512,3), np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if "Q8O" in text:
                if len(text)==11:
                    cv2.putText(img2,text,(10,500), font, 1,(0,153,0),2,cv2.LINE_AA)
                    cv2.putText(img2,"Pass",(80,80), font, 2,(0,153,0),3,cv2.LINE_AA)
                    totalpass+=1
                    cv2.putText(img2,str(totalpass),(10,80), font, 2,(0,153,0),3,cv2.LINE_AA)
                    cv2.imshow("Results",img2)
                    checkingSerial=False
                else:
                    cv2.putText(img2,text,(10,500), font, 1,(0,0,250),2,cv2.LINE_AA)
                    cv2.putText(img2,str(totalpass),(10,80), font, 2,(0,153,0),3,cv2.LINE_AA)
                    cv2.imshow("Results",img2)
                    checkingSerial=False
            else:
                cv2.putText(img2,text,(10,500), font, 1,(0,0,250),2,cv2.LINE_AA)
                cv2.putText(img2,str(totalpass),(10,80), font, 2,(0,153,0),3,cv2.LINE_AA)
                cv2.imshow("Results",img2)
                checkingSerial=False
        else :
          break
        if(printcount == count):
            break

        # Draw the results of the detection (aka 'visualize the results')
            #serialimg = None

    




    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        #cv2.imwrite("ignore.JPG", serialimg)
        break

# Clean up
video.release()
cv2.destroyAllWindows()
