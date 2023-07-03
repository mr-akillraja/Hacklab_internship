# # YOLO object detection
import cv2 
import numpy as np
import time

# img = cv.imread('horse.jpg')
# cv.imshow('window',  img)
# cv.waitKey(0)

# # Give the configuration and weight files for the model and load the network.
# net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# ln = net.getLayerNames()
# print(len(ln), ln)

# # construct a blob from the image
# blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
# r = blob[0, 0, :, :]

# cv.imshow('blob', r)
# text = f'Blob shape={blob.shape}'
# cv.displayOverlay('blob', text)
# cv.waitKey(0)

# net.setInput(blob)
# t0 = time.time()
# outputs = net.forward(ln)
# t = time.time()

# cv.displayOverlay('window', f'forward propagation time={t-t0}')
# cv.imshow('window',  img)
# cv.waitKey(0)


img = cv2.imread("C:\\Users\\Lenovo\\Documents\\py\\cv\\Track_Images\\tracking_2.jpeg")



# Load the YOLO network model from the harddisk into opencv
yolo = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Get the neural components of the YOLO model
ln = yolo.getLayerNames()
print(len(ln),ln)

# Creation of  a blob 
# Input to the network is called blob object.
# This function transforms the image into a blob.
blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB = True,crop = False)
r = blob[0,0,:,:]
cv2.imshow('blob',r)
cv2.waitKey(0)

yolo.setInput(blob)
t0 = time.time()
outputs = yolo.forward(ln)
t = time.time()

cv2.imshow("window",img)
cv2.waitKey(0)




cv2.imshow("orignal_IMg",img)

cv2.waitKey(0)