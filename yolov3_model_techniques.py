import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from techniques import img_paths,perform_data_augmentation


def perform_object_detection(images_folder, weights_file, config_file, classes_file):

    # Reading the darknet files
    yolo = cv2.dnn.readNet(weights_file, config_file)

    # coco.names : many classes.
    classes = []
    with open(classes_file,'r') as f:
        classes = f.read().splitlines()


    # Iterate over the files in the folder
    for filename in os.listdir(images_folder):
        if filename.endswith('.jpeg'):
            # Path to the image file
            image_path = os.path.join(images_folder, filename)
            
            # Image enhancement or image splitting
            
            # Read the image using OpenCV
            img = cv2.imread(image_path)
            (h,w) = img.shape[:2]

            center_x,center_y = (w//2),(h//2)

            # Top left of the image
            top_left = img[0:center_y,0:center_x]

            # Top right of the image
            top_right = img[0:center_y,center_x:w]
            bottom_left = img[center_y:h,0:center_x]

            # Bottom left of the image.

            # Bottom right of the image
            bottom_right = img[center_y:h,center_x:w]



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
                

images_folder = 'train_set/'
weights_file = 'yolov3-tiny.weights'
config_file = 'yolov3-tiny.cfg'
classes_file = 'coco.names'

# perform_object_detection(images_folder, weights_file, config_file, classes_file)

image_paths = img_paths('train_set')
output_dir = 'preview'
num_images = 2

for image_path in image_paths:
    # Perform data augmentation for each image path
    perform_data_augmentation(image_path, output_dir, num_images)
perform_object_detection(output_dir,weights_file,config_file,classes_file)

