# Object Detection Without Any Models
- Performing object detection without using any pre-trained model or specific object detection algorithm can be a challenging task to handle.
- Object Detection Algorithm are designed to learn and recognize patterns in images or videos,making it easier to identify and locate object of interest.
- If you want to explore a simple approach for object detection without using a pre-trained model,you can try the following the steps.


# Steps involved in the process
## STEP - 1
- Read the Image
## STEP - 2
- Convert the image into grayscale image
## STEP - 3
- use blurring effect in the grayscale image
## STEP - 4
- use edge detection methods to find the edges.
## STEP - 5
- Apply dilation and Erosion to close gaps in between the object edges.
## STEP - 6
- Find the contours of the objects
## STEP - 7
- Loop over the Detected Contours.
## STEP - 8
- Filter the Contours based on the area.
## STEP - 9
- Get the bounding box coordinates
## STEP - 10
- Draw the bounding box.
## STEP - 11
- Finally, Display the image.


# Advantages and Disadvantages
## Advantages ##
    - Simplicity
    - Speed
    - No Training Data Requirements
    - Customizability

## Disadvantages ##
    - Limited Accuracy
    - Lack of Generalization
    - Manual Feature Engineering
    - Limited Scalability
