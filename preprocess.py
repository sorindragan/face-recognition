import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# read and resize original image to orl database ratio
def resize():
    path = './att_faces/orl_faces/andreea/'

    for img in os.listdir(path):
        image = cv2.imread(path + img, cv2.IMREAD_GRAYSCALE)

        name = str(os.listdir(path).index(img)) + '.jpg'
        image  = cv2.resize(image, (112, 92))
        cv2.imwrite(name, image)

# read and save image as grayscale
def gary():
    path = './andreeaneagra/'

    for img in os.listdir(path):
        image = cv2.imread(path + img, cv2.IMREAD_GRAYSCALE)
        name = str(os.listdir(path).index(img)) + '.pgm'
        cv2.imwrite(name, image)
