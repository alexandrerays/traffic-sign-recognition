import os
import random
from sklearn.utils import shuffle
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
from time import time
import tensorflow as tf

import utils as utils

#Constants
img_size = 64
batch_size = 50
epoch_num = 500

#PATHS
ROOT_PATH = "/traffic"
train_data_dir = "datasets/BelgiumTS/Training"
test_data_dir = "datasets/BelgiumTS/Testing"

#Loads image data
images, labels = utils.load_data(train_data_dir)
test_images, test_labels = utils.load_data(train_data_dir)

#Prints
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

# Resize images
images_resized = [skimage.transform.resize(image, (img_size, img_size), mode='constant')
                for image in images]
test_images_resized = [skimage.transform.resize(image, (img_size, img_size), mode='constant')
                for image in test_images]

labels_a = np.array(labels)
images_a = np.array(images_resized)

# learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10]
# learning_rates = [0.05]
lr = 0.05
neurons_list = [5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000]
csv_array = np.empty([2*len(neurons_list), int(epoch_num/10)+4])

for j in range(len(neurons_list)):
    #Graph building
    graph = tf.Graph()
    with graph.as_default():
        images_ph = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
        labels_ph = tf.placeholder(tf.int32, [None])

        images_flat = tf.contrib.layers.flatten(images_ph)

        # conv1 = tf.contrib.layers.conv2d(images_ph, 16, 5, stride=1)
        # drop1 = tf.contrib.layers.dropout(conv1)
        # maxpool1 = tf.contrib.layers.max_pool2d(drop1, 2, stride=2)

        # conv_flat = tf.contrib.layers.flatten(conv6)

        layer1 = tf.contrib.layers.fully_connected(images_flat, neurons_list[j], tf.nn.tanh)
        logits = tf.contrib.layers.fully_connected(layer1, 62, tf.nn.tanh)

        predicted_labels = tf.argmax(logits, 1)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

        # learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

        init = tf.global_variables_initializer()

    session = tf.Session(graph=graph)

    #Preparing
    session.run([init])
    images_a, labels_a = shuffle(images_a, labels_a, random_state=0)
    csv_array[2*j][0] = neurons_list[j]
    csv_array[2*j+1][0] = neurons_list[j]
    csv_array[2*j][3] = 0
    csv_array[2*j+1][3] = 0

    #run all training epochs
    tempo_inicial = time()
    for i in range(epoch_num):
        _, loss_value = session.run([train, loss], feed_dict={images_ph: images_a, labels_ph: labels_a})
        if i % 10 == 9:
            predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_resized})[0]
            match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
            accuracy = match_count / len(test_labels)
            print("Loss: {:.3f}      Accuracy: {:.3f}".format(loss_value, accuracy))
            csv_array[2*j][int(i/10)+4] = loss_value
            csv_array[2*j+1][int(i/10)+4] = accuracy
    tempo_final = time()
    tempo_execucao = tempo_final - tempo_inicial
    print('Tempo de Training:', tempo_execucao)
    csv_array[2*j][1] = tempo_execucao
    csv_array[2*j+1][1] = tempo_execucao

    #run test data
    tempo_inicial = time()
    predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_resized})[0]
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))
    tempo_final = time()
    tempo_execucao = tempo_final - tempo_inicial
    print('Tempo de processamento:', tempo_execucao)
    csv_array[2*j][2] = tempo_execucao
    csv_array[2*j+1][2] = tempo_execucao

    np.savetxt("neurons_list2.csv", csv_array.T, delimiter=",", fmt='%10.5f')
    session.close()