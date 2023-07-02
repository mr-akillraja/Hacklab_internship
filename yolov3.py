import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Reading the darknet files
yolo = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")

# coco.names : many classes.
classes = []
with open('coco.names','r') as f:
  classes = f.read().splitlines()


# Folder path containing the images
folder_path = 'images/'

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg'):
        # Path to the image file
        image_path = os.path.join(folder_path, filename)

        # Read the image using OpenCV
        img = cv2.imread(image_path)

    height,width,channels = img.shape
    # Convert this to RGB Image
    # It is 4 dimension array pre-processed image.
    blob = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False)

    # Module importing from outside
    yolo.setInput(blob)
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layerout = yolo.forward(output_layer_name)


    # find the bounding boxes and put that on the image.
    boxs = []
    # at confidence it is been predicted
    confidences = []
    # to get what class it is
    class_ids = []

    for output in layerout:
        for detection in output:
            # Getting a list
            # Score is used to getting max probability of image as a array.
            score = detection[5:]
            # get the percentage of the classes.
            class_id = np.argmax(score)
            confidence = score[class_id]
            if  confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[0]*height)
                w = int(detection[0]*width)
                h = int(detection[0]* height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxs.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(img, f'{classes[class_id]}: {confidence:.2f}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(classes[class_id])

                cv2.imshow("image",img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()