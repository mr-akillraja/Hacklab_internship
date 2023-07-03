import tensorflow as tf
import cv2
import os
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split


def load_class_names(filename):
    with open(filename, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def img_paths(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpeg'):
            # Path to the image file
            image_path = os.path.join(folder_path, filename)
            images.append(image_path)
    return images

def darknet_convolutional(inputs, filters, kernel_size, strides=1, padding='same'):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def darknet_residual(inputs, filters):
    x = darknet_convolutional(inputs, filters // 2, 1)
    x = darknet_convolutional(x, filters, 3)
    x = tf.keras.layers.Add()([inputs, x])
    return x

def darknet_block(inputs, filters, num_blocks):
    x = darknet_convolutional(inputs, filters, 3, strides=2)
    for _ in range(num_blocks):
        x = darknet_residual(x, filters)
    return x

def yolo_v3(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Backbone
    x = darknet_convolutional(inputs, 32, 3)
    x = darknet_block(x, 64, num_blocks=1)
    x = darknet_block(x, 128, num_blocks=2)
    x = darknet_block(x, 256, num_blocks=8)
    x = darknet_block(x, 512, num_blocks=8)
    x = darknet_block(x, 1024, num_blocks=4)

    # Head
    x = darknet_convolutional(x, 512, 1)
    x = darknet_convolutional(x, 1024, 3)
    x = darknet_convolutional(x, 512, 1)
    x = darknet_convolutional(x, 1024, 3)
    x = darknet_convolutional(x, 512, 1)

    # Detection layers
    output_1 = darknet_convolutional(x, 1024, 3)
    output_1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(output_1)

    x = darknet_convolutional(x, 256, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, darknet_convolutional(x, 512, 3)])

    output_2 = darknet_convolutional(x, 512, 3)
    output_2 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(output_2)

    x = darknet_convolutional(x, 128, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, darknet_convolutional(x, 256, 3)])

    output_3 = darknet_convolutional(x, 256, 3)
    output_3 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(output_3)

    model = tf.keras.Model(inputs=inputs, outputs=[output_1, output_2, output_3])

    return model

# Create the YOLOv3 model with input shape (416, 416, 3) and 80 classes (COCO dataset)
# Load the class names
class_names = load_class_names('coco.names')

# Prepare the training data
image_paths = img_paths('train_set')
num_classes = len(class_names)

# Create empty lists for input images and labels
images = []
labels = []

# Iterate over the image paths
for image_path in image_paths:
    # Load and preprocess the image
    image = cv2.imread(image_path)
    images.append(image)

    

# Convert the lists to NumPy arrays
images = np.array(images)


# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2)

# Define the model
model = yolo_v3((416, 416, 3), num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(val_images, val_labels))
