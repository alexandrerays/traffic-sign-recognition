import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf


#Loads image data
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

#Display images
def display_first_image_each_label(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(10, 10))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.show()

#Display images from same label
def display_images_in_label(images, labels, count = 10, target_label = 0):
    plt.figure(figsize=(10, 10))
    i = 1
    l_pointer = 0
    breaker = 0
    for label in labels:
        if label == target_label:
            image = images[l_pointer]
            plt.subplot(10, 10, i)  # A grid of 8 rows x 8 columns
            plt.axis('off')
            plt.title("Label {0} ({1})".format(label, labels.count(label)))
            i += 1
            plt.imshow(image)
            breaker += 1
            if breaker >= count:
                break
        l_pointer += 1
    plt.show()

# Display the predictions and the ground truth visually.
def display_predicted_images(images, labels, predicted):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        truth = labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
                 fontsize=12, color=color)
        plt.imshow(images[i])
    plt.show()