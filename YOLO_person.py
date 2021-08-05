#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PERSON DETECTOR
Created on Mon Aug  2 12:59:56 2021

@author: peterriley
"""
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import glob
import os
import argparse
import time 
from timeit import default_timer as timer


parser = argparse.ArgumentParser(description='A function to blur faces in images in a folder and save them in a new folder.')
parser.add_argument("-i","--input", help="Path to image folder. Make sure it only contains images. NOTE, end this line with a '/' after the folder name. No * needed ")
parser.add_argument("-o","--output", help="Path to where the new images are to be saved")
parser.add_argument("-n","--name", help="optional argument. New tag of the the saved images (i.e. tag + original name)", default = "new")




args = parser.parse_args()
start_time = time.time()
start = timer()



whT = 320
confThreshold =0.5
nmsThreshold= 0.2
print("Confidence level threshold is:", confThreshold)
#### LOAD MODEL
## Coco Names
classesFile = "Yolo wieghts and names/coco.names"
classNames = []
with open(classesFile, "rt") as f:
    classNames=f.read().strip('\n').split('\n')
#print(classNames)
## Model Files
modelConfiguration = "Yolo wieghts and names/yolov3-tiny.cfg"
modelWeights = "Yolo wieghts and names/yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def findObjects(outputs,img, counter):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
# =============================================================================
#     t2 = time.clock() - t1
#     print("Time elapsed to find stuff: ", t2)
# =============================================================================
    value = 0
    if 0 in indices:
        for i in indices:
            i = i[0]
            
           
            if classIds[i] in [0]: #classIds refering to people
                box = bbox[i]
              
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
# =============================================================================
#                 cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
#                 cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
#                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
# =============================================================================
                value = 1
                counter = counter +1

        return img, value, counter
    else:

        img2 = img
        return img2, value, counter
        

def rect_to_ellipse(x, y, width,  height):
    vert_axis  = round(height/2)
    horz_axis = round(width/2)
    center_x = round(x + horz_axis)
    center_y =  round(y +  vert_axis)
    center_coordinates = (center_x, center_y)
    axesLength = (horz_axis, vert_axis)    
    return center_coordinates, axesLength

# note: 15 is the local window - this is arbitrary and we may change it later
def blur(image):
    blurred = cv2.medianBlur(image, 15)
    return blurred

def face_detect(image):
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(image)
    return  faces    

def load_image(file):
    image = cv2.imread(file)
    #convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def logical_mask(image, scrambled, mask):
    fg = cv2.bitwise_or(scrambled, scrambled, mask=mask)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(image, image, mask=mask)
    newImage = cv2.bitwise_or(fg, bk)
    return newImage

# create list of images
imageList =glob.glob(args.input+"*") #glob.glob('/Users/peterriley/Desktop/Blur/*') # need to put pathway to images here​
#path = '/Users/peterriley/Desktop/NewImages/'             # this provides a path to whatever desired location to save the blurred images
path = args.output
def find_people(imageList, path, name):
    
    # main loop
    counter = 0
    tracker = 0
    for i in range(len(imageList)):
        # load image
        img = load_image(imageList[i])
        
        
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        img, value, counter = findObjects(outputs,img, counter)

        
        
        
        
        
        
        
        
        
        
        
# =============================================================================
#         
#         
#         # detect faces in image
#         faceCoordinates = face_detect(image)
#         
#         # create blurred version of entire image
#         scrambled = blur(image)
#         
#         # create mask of zeros
#         mask = np.full((scrambled.shape[0], scrambled.shape[1]), 0, dtype=np.uint8)
#         
#         # for each face, convert bounding box to ellipse
#         for j in range(len(faceCoordinates)): #j = which face in the frame
#             x,y, width, height = (faceCoordinates[j]['box'])
#             #converts the bounding box to an ellipse via a custom function
#             ellipse = rect_to_ellipse(x, y, width,  height)
#             #puts the ellipse onto the mask
#             cv2.ellipse(mask, ellipse[0], ellipse[1], 0, 0, 360, 255, -1)
#         
#         # apply logical masking to each face
#         newImage = logical_mask(image, scrambled, mask)
# =============================================================================

        if value ==1:
            tracker = tracker +1
        if tracker%100 ==0:
            print("Don't freak out. Still processing. Images with people detected:", tracker)
# =============================================================================
#             newImage  = img
#             # write new image to disk
#             newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
#             basename = os.path.basename(imageList[i])     
#             cv2.imwrite(os.path.join(path + name +" "+ basename), newImage) 
# =============================================================================
    print("total number of images with people: ", tracker )
    print("People found:", counter)
    
find_people(imageList, path, args.name)    
print("--- %s seconds ---" % (time.time() - start_time))
end = timer()
print("This is the timer version",end - start) # Time in seconds, e.g. 5.38091952400282
print("Threshold:", confThreshold)
