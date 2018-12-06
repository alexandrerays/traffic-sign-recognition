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
epoch_num = 1000

#PATHS
ROOT_PATH = "/traffic"
train_data_dir = "datasets/BelgiumTS/Training"
test_data_dir = "datasets/BelgiumTS/Testing"

#Loads image data
images, labels = utils.load_data(train_data_dir)
test_images, test_labels = utils.load_data(test_data_dir)

#Prints
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

# Resize images
images_resized = [skimage.transform.resize(image, (img_size, img_size), mode='constant')
                for image in images]
test_images_resized = [skimage.transform.resize(image, (img_size, img_size), mode='constant')
                for image in test_images]

labels_a = np.array(labels)
images_a = np.array(images_resized)

csv_array = np.empty([4, int(epoch_num/10)+4])

csv_array[0][0] = 0
csv_array[0][1] = 0
csv_array[0][2] = 0
csv_array[1][0] = 0

#Graph building
graph = tf.Graph()
with graph.as_default():
    images_ph = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # images_flat = tf.contrib.layers.flatten(images_ph)

    conv1 = tf.contrib.layers.conv2d(images_ph, 16, 5, stride=1)
    drop1 = tf.contrib.layers.dropout(conv1)
    # conv2 = tf.contrib.layers.conv2d(drop1, 8, 5, stride=1)
    # drop2 = tf.contrib.layers.dropout(conv2)
    # maxpool1 = tf.contrib.layers.max_pool2d(drop1, 2, stride=2)

    conv_flat = tf.contrib.layers.flatten(drop1)

    layer1 = tf.contrib.layers.fully_connected(conv_flat, 250, tf.nn.tanh)
    logits = tf.contrib.layers.fully_connected(layer1, 62, tf.nn.tanh)
    softmax = tf.nn.softmax(logits)

    predicted_labels = tf.argmax(logits, 1)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
    train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    init = tf.global_variables_initializer()

session = tf.Session(graph=graph)

#Preparing
session.run([init])
images_a, labels_a = shuffle(images_a, labels_a, random_state=0)

#run all training epochs
tempo_inicial = time()
for i in range(epoch_num):
    _, loss_value = session.run([train, loss], feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 100 == 99:
    	print("Loss: {:.3f}".format(loss_value))
    # if i % 50 == 49:
        # predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_resized})[0]
        # match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
        # accuracy = match_count / len(test_labels)
        # predicted = session.run([predicted_labels], feed_dict={images_ph: images_resized})[0]
        # match_count = sum([int(y == y_) for y, y_ in zip(labels, predicted)])
        # accuracy_train = match_count / len(labels)
        # print("Loss: {:.3f}      Accuracy: {:.3f}      TAccuracy: {:.3f}".format(loss_value, accuracy, accuracy_train))
        # csv_array[0][int(i/50)+4] = i
        # csv_array[1][int(i/50)+4] = loss_value
        # csv_array[2][int(i/50)+4] = accuracy
        # csv_array[3][int(i/50)+4] = accuracy_train
        # if i % 1000 == 999:
        #     predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_resized})[0]
        #     prediction_csv = np.empty([2, len(test_labels)])
        #     prediction_csv[0] = test_labels
        #     prediction_csv[1] = predicted
        #     np.savetxt("predictions4_" + str(i) + "_test.csv", prediction_csv.T, delimiter=",", fmt='%10.5f')
        #     predicted_train = session.run([predicted_labels], feed_dict={images_ph: images_resized})[0]
        #     prediction_train_csv = np.empty([2, len(labels)])
        #     prediction_train_csv[0] = labels
        #     prediction_train_csv[1] = predicted_train
        #     np.savetxt("predictions4_" + str(i) + "_train.csv", prediction_train_csv.T, delimiter=",", fmt='%10.5f')
tempo_final = time()
tempo_treinamento = tempo_final - tempo_inicial
print('Tempo de Training:', tempo_treinamento)
csv_array[1][1] = tempo_treinamento

#run test data
tempo_inicial = time()
predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_resized})[0]
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
tempo_final = time()
tempo_execucao = tempo_final - tempo_inicial
print('Tempo de processamento:', tempo_execucao)
csv_array[1][2] = tempo_execucao


soft = session.run([softmax], feed_dict={images_ph: test_images_resized})[0]
predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_resized})[0]
prediction_csv = np.empty([3, len(test_labels)])
prediction_csv[0] = test_labels
prediction_csv[1] = predicted
for j in range(len(test_labels)):
	prediction_csv[2][j] = soft[j][predicted[j]]
np.savetxt("predictions_and_probabilites.csv", prediction_csv.T, delimiter=",", fmt='%10.5f')
np.savetxt("soft.csv", soft.T, delimiter=",", fmt='%10.5f')

np.savetxt("dsnet5.csv", csv_array.T, delimiter=",", fmt='%10.5f')
session.close()