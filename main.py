import os
import random
from sklearn.utils import shuffle
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils as utils

#Constants
img_size = 32
batch_size = 500
epoch_num = 201

#PATHS
ROOT_PATH = "/traffic"
train_data_dir = "datasets/BelgiumTS/Training"
test_data_dir = "datasets/BelgiumTS/Testing"

#Loads image data
images, labels = utils.load_data(train_data_dir)
test_images, test_labels = utils.load_data(train_data_dir)

#Prints
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

#utils.display_first_image_each_label(images, labels)
#utils.display_images_in_label(images, labels, count = 20, target_label = 26)
#for image in images[:5]:
#    print("shape: {0}, min: {1}, max: {2}".format(
#          image.shape, image.min(), image.max()))

# Resize images
images_resized = [skimage.transform.resize(image, (img_size, img_size), mode='constant')
                for image in images]
test_images_resized = [skimage.transform.resize(image, (img_size, img_size), mode='constant')
                for image in test_images]

#utils.display_first_image_each_label(images_resized, labels)
#for image in images_resized[:5]:
#    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


#Graph building
graph = tf.Graph()
with graph.as_default():
    images_ph = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    conv1 = tf.contrib.layers.conv2d(images_ph, 16, 5)
    drop1 = tf.contrib.layers.dropout(conv1)
    maxpool1 = tf.contrib.layers.max_pool2d(drop1, 2, stride=2)

    # conv2 = tf.contrib.layers.conv2d(maxpool1, 16, 5)
    # drop2 = tf.contrib.layers.dropout(conv2)
    # maxpool2 = tf.contrib.layers.max_pool2d(drop2, 2, stride=2)

    conv_flat = tf.contrib.layers.flatten(maxpool1)

    # Fully connected layer. 
    # Generates logits of size [None, 62]
    layer1 = tf.contrib.layers.fully_connected(conv_flat, 62, tf.nn.relu)

    # Fully connected layer.    
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(layer1, 62,
        tf.nn.relu)

    # Convert logits to label indexes.
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

# print("images_flat: ", images_flat)
# print("logits: ", logits)
# print("loss: ", loss)
# print("predicted_labels: ", predicted_labels)

# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
session.run([init])

labels_a = np.array(labels)
images_a = np.array(images_resized)

images_a, labels_a = shuffle(images_a, labels_a, random_state=0)

for i in range(epoch_num):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)

# for i in range(epoch_num):
#     loss_sum = 0
#     for j in range(int(len(images_a)/batch_size)):
#         labels_batch = labels_a[j*batch_size:(j+1)*batch_size] 
#         images_batch = images_a[j*batch_size:(j+1)*batch_size]
#         _, loss_value = session.run([train, loss], 
#                                     feed_dict={images_ph: images_batch, labels_ph: labels_batch})
#         loss_sum += loss_value
#     if i % 10 == 0:
#         print("Loss: ", loss_sum/j)

# Run prediction against test data
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images_resized})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))

#print(sample_labels)
#print(predicted)
#utils.display_predicted_images(sample_images, sample_labels, predicted)

# Close the session. This will destroy the trained model.
session.close()