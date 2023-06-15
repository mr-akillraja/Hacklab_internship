# Object Detection without any technique used 


import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image



# Reading the Image
img = cv2.imread("C:\\Users\\Lenovo\\Documents\\py\\cv\\Track_Images\\tracking_11.jpeg")

# converting original image to grayscale image
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# conversion of grayscale image into blurring image to reduce noise
img_blur = cv2.GaussianBlur(img_gray,(5,5),0)

# conversion of blurring image to find the edges in the image.
img_edges = cv2.Canny(img_blur,50,150)

# after finding edges making the image dilated
dilated_img = cv2.dilate(img_edges,None,iterations=2)

# after dilation of the image then it is been eroded
erode_img = cv2.erode(dilated_img,None,iterations=1)

# finding the boundaries of objects and shapes in an image..
img_countour,_ = cv2.findContours(erode_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


# Looping over the detected contours
for countour in img_countour:
    # Filter contours based on the area
    if cv2.contourArea(countour) > 100:
        # Get the Bounding Box coordinates
        x,y,w,h = cv2.boundingRect(countour)

        # Draw the bounding box
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# Display of the image

cv2.imshow("Object_Detection",img)
cv2.waitKey(0)