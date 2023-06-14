# YOLO #
### full form ###
# You Only Look Once #


## Oject Localization 
- Here , where are  defining class and also defining where the object is been located in the image.
- It will be having some matrix format for locating the different classes of object.
- The matrix consists of Pc(probablity of any class),B(x,y,w,n) are the boundary points of the box which is been used for detection of object and classification,C(1,2) tells which class it belongs to 1 or 2.
- The x_train is an image as input and the y_train is the matrix defined above.
- Sending these features to the neural network for training the model for classification of image with detection of object.
- Since it is supervised learning , we have to define y_train individually for each image.
- This works only for single object.

## What about Multiple Objects in an image ?
- For example : if you are having a image with two objects or bounding boxes.
- Then YOLO algorithm will divide the image into grid cells (3X3),(5X5)
- Taking each cell and detecting whether the object is detected in the grid cells of the image.
- If the object is detected in the image then findings its center of the object in the divided grid cells.
- It uses a method called Intersection Over unions or (IOU).

## IOU  [ Intersection Over Unions ] ## 
- Most of the time , a single object in an image can have multiple grid box candidates for prediction, even though not all them are relevant.
- IOU =  Intersection Area / Union Area
- The goal of the IOU is to discard such grid boxes to only keep those that are relevant.
- Logic :
    - User defines its IOU selection threshold , which can be , for instance 0.5
    - Then YOLO computes the IOU of each grid cell which is the Intersection area divided by the union area.
    - Finally, it ignores the prediction of the grid cells having an IOU <= Threshold and considers those with an IOU > Threshold.

## Non-Max Supression or NMS
- Setting a threshold for the IOU is not always enough because an object can have multiple boxes with IOU beyond the Threshold,and leaving all those boxes might include noise.
- Here is where we can use NMS to keep only the boxes with the highest probability score of detection.

## What if one grid cell has center of two ojects?
- The boundary matrix will have two matrix with different centers for two different classes.
- Here , We are jsut concatenating both boundary matrix of the two objects in the image.
- finally we are having 14 dimensions vector.
- This concept is called anchor boxes.
- You can have more anchor boxes . 

# YOLO Application 
- Application in Industries
    - Healthcare
    - Agriculture
- Security Surveillance
- Self-Driving Cars

# Types of YOLO algorithm
- From year to year there will be addition of feature for increasing the accuracy and to control the error rates.
- Types : 
    - YOLO
    - YOLOv2 or YOLO9000
    - YOLOv4
    - YOLOR
    - YOLOx
    - YOLOv5
    - YOLOv6
    - YOLOv7

# What makes YOLO popular for Object Detecion ?
- Spedd
- Detection Accuracy 
- Good generalization
- Open-Source
  