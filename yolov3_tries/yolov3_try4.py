import tensorflow as tf
from auto_encoder import train_autoencoder
import os
import cv2


def preprocess_image(image_path):
    # Load the image using TensorFlow
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize the image to the model's input shape
    image = tf.image.resize(image, [416, 416])

    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0

    # Apply data augmentation (e.g., random flip, random rotation)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    return image


def load_and_preprocess_dataset(dataset_path, batch_size):
    # Get the list of image file paths
    image_files = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.jpeg')]
    # Create a dataset from the list of image file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    # Load and preprocess the images in parallel
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(image_files)).batch(batch_size)
    # Prefetch the next batch for better performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

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

def create_yolo_model(input_shape, num_classes):
    # Input layer
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
    output1 = darknet_convolutional(x, 1024, 3)
    output1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(output1)

    x = darknet_convolutional(x, 256, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, darknet_convolutional(x, 512, 3)])

    output2 = darknet_convolutional(x, 512, 3)
    output2 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(output2)

    x = darknet_convolutional(x, 128, 1)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, darknet_convolutional(x, 256, 3)])

    output3 = darknet_convolutional(x, 256, 3)
    output3 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)(output3)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=[output1, output2, output3])

    return model

# Create the YOLOv3 model with input shape (416, 416, 3) and 80 classes (COCO dataset)
model = create_yolo_model((416, 416, 3), 80)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# Specify the path to your training dataset
train_dataset_path = 'train_set/'

image_folder = 'train_set'  # Replace with the path to your image folder
input_shape = (256, 256, 3)  # Adjust according to your image size and channels
latent_dim = 128  # Adjust according to your desired latent space dimension
batch_size = 32
epochs = 10

train_dataset = load_and_preprocess_dataset(train_dataset_path, batch_size)
train_labels = train_autoencoder(image_folder, input_shape, latent_dim, batch_size, epochs)

model.fit(train_dataset,train_labels,batch_size=batch_size, epochs=epochs)


test_dataset = ...  # Load your test dataset
model.evaluate(test_dataset)
