from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
import os
import cv2

def img_paths(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpeg'):
            # Path to the image file
            image_path = os.path.join(folder_path, filename)
            images.append(image_path)
    return images

def perform_data_augmentation(image_path, output_dir, num_images=20):
    # Create an instance of ImageDataGenerator with desired augmentation parameters
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Load the image
    img = load_img(image_path)
    x = img_to_array(img)

    # Reshape the image array
    x = x.reshape((1,) + x.shape)

    # Generate augmented images and save them to the output directory
    i = 0
    for batch in data_gen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='data_aug', save_format='jpeg'):
        i += 1
        if i >= num_images:
            break
    return output_dir

image_paths = img_paths('train_set/')
output_dir = 'preview'
num_images = 2

for image_path in image_paths:
    # Perform data augmentation for each image path
    perform_data_augmentation(image_path, output_dir, num_images)
