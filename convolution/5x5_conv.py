
import torch
import numpy as np
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageOps

# Using torch library for gpu activation

def show_image(m, color):
    if color == 'red':
        m = Image.fromarray(np.asarray(m), 'L')
        m = ImageOps.colorize(m, black='black', white='red')
    # if color == 'green':
    #     m = Image.fromarray(np.asarray(m), 'L')
    #     m = ImageOps.colorize(m, black='black', white='green')
    if color == 'blue':
        m = Image.fromarray(np.asarray(m), 'L')
        m = ImageOps.colorize(m, black='black', white='blue')
    if color == 'rgb':
        m = Image.fromarray(np.asarray(m), 'RGB')
    m.show()


def rgb_convolution(im, filt):

    # # create three separate images
    #
    # red_picture, green_picture, blue_picture = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    #
    # red_picture = red_picture.view(1, 1, len(red_picture), len(red_picture[0])).byte()
    # green_picture = green_picture.view(1, 1, len(green_picture), len(green_picture[0])).byte()
    # blue_picture = blue_picture.view(1, 1, len(blue_picture), len(blue_picture[0])).byte()
    #
    # red_features = F.conv2d(torch.tensor(red_picture), filt, padding=1)
    # green_features = F.conv2d(torch.tensor(green_picture), filt, padding=1)
    # blue_features = F.conv2d(torch.tensor(blue_picture), filt, padding=1)
    #
    # # turn back into array with two axis
    # red_features = np.asarray(red_features.squeeze().squeeze())
    # green_features = np.asarray(green_features.squeeze().squeeze())
    # blue_features = np.asarray(blue_features.squeeze().squeeze())
    #
    # return red_features, green_features, blue_features

    # create one rgb image

    # RGB BELOW

    im = im.view(1, 3, len(im), len(im[0])).byte()
    im_features = F.conv2d(torch.tensor(im), filt, padding=1)
    im_features.squeeze()

    return im_features


def convolute_image(im, filt):
    # show_image(im, 'RGB')
    red_features, blue_features, green_features = rgb_convolution(im, filt)
    # show_image(red_features, 'red')
    show_image(blue_features, 'green')
    # show_image(green_features, 'blue')


# 5x5 filter
random_filter = torch.tensor([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])



random_filter = random_filter.view(1, 1, 5, 5).byte()

# random_filter = torch.tensor([1, 1, random_filter])

# random_filter.view(1,1,5,5).repeat(1, 1, 1, 1)

im = Image.open('../img/ct.jpg')
im = torch.tensor((np.array(im)))
im = im.clone()


# 5x5 filter
random_filter = torch.tensor([[0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])


random_filter = random_filter.view(1, 1, 5, 5).byte()
random_filter = random_filter.repeat(3, 3, 1, 1)
im = rgb_convolution(im, random_filter)
im = im.view(1078, 1614, 3)
im = np.array(im)

show_image(im, 'rgb')















# TO DELETE

# separate each pixel into red, green, blue

# red_pixels = torch.tensor([[im[i-2][j-2][0], im[i-2][j-1][0], im[i-2][j][0], im[i-2][j+1][0], im[i-2][j+2][0]],
#               [im[i-1][j-2][0], im[i-1][j-1][0], im[i-1][j][0], im[i-1][j+1][0], im[i-1][j+2][0]],
#               [im[i][j-2][0], im[i][j-1][0], im[i][j][0], im[i][j+1][0], im[i][j+2][0]],
#               [im[i+1][j-2][0], im[i+1][j-1][0], im[i+1][j][0], im[i+1][j+1][0], im[i+1][j+2][0]],
#               [im[i+2][j-2][0], im[i+2][j-1][0], im[i+2][j][0], im[i+2][j+1][0], im[i+2][j+2][0]]])
#
# green_pixels = torch.tensor([[im[i-2][j-2][1], im[i-2][j-1][1], im[i-2][j][1], im[i-2][j+1][1], im[i-2][j+2][1]],
#               [im[i-1][j-2][1], im[i-1][j-1][1], im[i-1][j][1], im[i-1][j+1][1], im[i-1][j+2][1]],
#               [im[i][j-2][1], im[i][j-1][1], im[i][j][1], im[i][j+1][1], im[i][j+2][1]],
#               [im[i+1][j-2][1], im[i+1][j-1][1], im[i+1][j][1], im[i+1][j+1][1], im[i+1][j+2][1]],
#               [im[i+2][j-2][1], im[i+2][j-1][1], im[i+2][j][1], im[i+2][j+1][1], im[i+2][j+2][1]]])
#
# blue_pixels = torch.tensor([[im[i-2][j-2][2], im[i-2][j-1][2], im[i-2][j][2], im[i-2][j+1][2], im[i-2][j+2][2]],
#               [im[i-1][j-2][2], im[i-1][j-1][2], im[i-1][j][2], im[i-1][j+1][2], im[i-1][j+2][2]],
#               [im[i][j-2][2], im[i][j-1][2], im[i][j][2], im[i][j+1][2], im[i][j+2][2]],
#               [im[i+1][j-2][2], im[i+1][j-1][2], im[i+1][j][2], im[i+1][j+1][2], im[i+1][j+2][2]],
#               [im[i+2][j-2][2], im[i+2][j-1][2], im[i+2][j][2], im[i+2][j+1][2], im[i+2][j+2][2]]])

# # move to gpu if possible
# red_pixels = red_pixels.to(device)
# green_pixels = green_pixels.to(device)
# blue_pixels = blue_pixels.to(device)
#
# red_pixel = torch.sum(torch.multiply(red_pixels, filt))#.astype(np.uint8)
# green_pixel = torch.sum(torch.multiply(green_pixels, filt))#.astype(np.uint8)
# blue_pixel = torch.sum(torch.multiply(blue_pixels, filt))#.astype(np.uint8)
#
# red_row.append(red_pixel)
# green_row.append(green_pixel)
# blue_row.append(blue_pixel)
#
# new_pixel = [red_pixel, green_pixel, blue_pixel]
# rgb_row.append(new_pixel)