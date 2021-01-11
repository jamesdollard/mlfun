
import torch
import numpy as np
import torchvision as tv
from PIL import Image

cifar_train = tv.datasets.CIFAR10('../data/CIFAR_10/train', train=True, transform=tv.transforms.PILToTensor, download=False)
cifar_test = tv.datasets.CIFAR10('../data/CIFAR_10/test', train=False, transform=tv.transforms.PILToTensor, download=False)

train_data = cifar_train.data


def show_image(m, type):
    Image.fromarray(m, type).show()


# define conv filter
horizontal_filter = [[1, 1, 1],
                     [0, 0, 0],
                     [-1, -1, -1]]
vertical_filter = [[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]]
random_filter = [[-1, -2, 0],
                 [-1, 0, 1],
                 [0, 2, 1]]

def rgb_convolution(im, filt):
    image_height = len(im)
    image_width = len(im[0])
    # apply filter for each pixel
    rgb_features = []
    red_features = []
    blue_features = []
    green_features = []
    for i in range(1, image_height - 1):
        red_row = []
        green_row = []
        blue_row = []
        rgb_row = []
        for j in range(1, image_width - 1):

            # separate each pixel into red, green, blue

            red_pixels = [[im[i-1][j-1][0], im[i-1][j][0], im[i-1][j+1][0]],
                          [im[i][j-1][0], im[i][j][0], im[i][j+1][0]],
                          [im[i+1][j-1][0], im[i+1][j][0], im[i+1][j+1][0]]]

            green_pixels = [[im[i-1][j-1][1], im[i-1][j][1], im[i-1][j+1][1]],
                            [im[i][j-1][1], im[i][j][1], im[i][j+1][1]],
                            [im[i+1][j-1][1], im[i+1][j][1], im[i+1][j+1][1]]]

            blue_pixels = [[im[i-1][j-1][2], im[i-1][j][2], im[i-1][j+1][2]],
                           [im[i][j-1][2], im[i][j][2], im[i][j+1][2]],
                           [im[i+1][j-1][2], im[i+1][j][2], im[i+1][j+1][2]]]

            red_pixel = np.sum(np.multiply(red_pixels, filt)).astype(np.uint8)
            green_pixel = np.sum(np.multiply(green_pixels, filt)).astype(np.uint8)
            blue_pixel = np.sum(np.multiply(blue_pixels, filt)).astype(np.uint8)

            red_row.append(red_pixel)
            green_row.append(green_pixel)
            blue_row.append(blue_pixel)

            new_pixel = [red_pixel, green_pixel, blue_pixel]
            rgb_row.append(new_pixel)

        red_features.append(red_row)
        green_features.append(green_row)
        blue_features.append(blue_row)
        rgb_features.append(rgb_row)

    rgb_features = np.asarray(rgb_features)
    red_features = np.asarray(red_features)
    green_features = np.asarray(green_features)
    blue_features = np.asarray(blue_features)

    return rgb_features, red_features, blue_features, green_features


def convolute_image(im, filt):
    show_image(im, 'RGB')
    rgb_image, red_features, blue_features, green_features = rgb_convolution(im, filt)
    show_image(rgb_image, 'RGB')
    show_image(red_features, 'P')
    show_image(blue_features, 'P')
    show_image(green_features, 'P')


# Convolute image from CIFAR-10 dataset
id = 0
convolute_image(train_data[id], horizontal_filter)

# Convolute own image
im = Image.open('../data/images/ct.jpg')
im = np.asarray(im)
convolute_image(im, horizontal_filter)
