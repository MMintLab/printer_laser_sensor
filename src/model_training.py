import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

dataset_folder = '/Users/william/Documents/MMINT_Research/printer_laser_sensor/dataset/'

#get sorted list of filenames beginning with 'input'
input_filenames = sorted([img for img in os.listdir(dataset_folder) if img.startswith('input')])
#get sorted list of filenames beginning with 'output'
output_filenames = sorted([img for img in os.listdir(dataset_folder) if img.startswith('output')])

#get weighted average of 24 arrays
def get_weighted_average(weights, array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, array13, array14, array15, array16, array17, array18, array19, array20, array21, array22, array23, array24):
    weighted_average = (weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4 + weights[4,0]*array5 + weights[5,0]*array6 + weights[6,0]*array7 + weights[7,0]*array8 + weights[8,0]*array9 + weights[9,0]*array10 + weights[10,0]*array11 + weights[11,0]*array12 + weights[12,0]*array13 + weights[13,0]*array14 + weights[14,0]*array15 + weights[15,0]*array16 + weights[16,0]*array17 + weights[17,0]*array18 + weights[18,0]*array19 + weights[19,0]*array20 + weights[20,0]*array21 + weights[21,0]*array22 + weights[22,0]*array23 + weights[23,0]*array24)
    return weighted_average

#compute cost given two arrays
def compute_cost(array1, array2):
    array1x = array1[:,0]
    array2x = array2[:,0]
    cost = np.sum(np.square(array1x - array2x))
    return cost

def get_highest_intensity_pixel_indices(image):
    highest_intensity_pixel_indices = np.zeros((image.shape[1], 2)).astype(int)
    for i in range(image.shape[1]):
        highest_intensity_pixel_indices[i,0] = np.argmax(image[:,i])
        highest_intensity_pixel_indices[i,1] = i
    return highest_intensity_pixel_indices

def get_red_channnel(image):
    red_channel = image[:,:,2]
    return red_channel

def get_green_channel(image):
    green_channel = image[:,:,1]
    return green_channel

def get_blue_channel(image):
    blue_channel = image[:,:,0]
    return blue_channel

def get_saturation_channel(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_channel = image[:,:,1]
    return saturation_channel

def get_value_channel(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = image[:,:,2]
    return value_channel

def get_hue_channel(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = image[:,:,0]
    return hue_channel

#shift image up one pixel
def shift_image_up(image):
    height, width = image.shape
    shifted_image = np.zeros((height, width), np.uint8)
    for i in range(0, height-1):
        for j in range(0, width):
            shifted_image[i][j] = image[i+1][j]
    return shifted_image

#shift image down one pixel
def shift_image_down(image):
    height, width = image.shape
    shifted_image = np.zeros((height, width), np.uint8)
    for i in range(1, height):
        for j in range(0, width):
            shifted_image[i][j] = image[i-1][j]
    return shifted_image

#shift image left one pixel
def shift_image_left(image):
    height, width = image.shape
    shifted_image = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(1, width):
            shifted_image[i][j] = image[i][j-1]
    return shifted_image

#shift image right one pixel
def shift_image_right(image):
    height, width = image.shape
    shifted_image = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width-1):
            shifted_image[i][j] = image[i][j+1]
    return shifted_image

#create 24x1 numpy array of ones
weights = np.ones((24,1))

numfiles = len(input_filenames)
prevcost = 2000
cost = 1000

incr = 0.001

learning_rate = 0.0001

gradient_shifted_up_red_channel_total = 0
gradient_shifted_down_red_channel_total = 0
gradient_shifted_left_red_channel_total = 0
gradient_shifted_right_red_channel_total = 0
gradient_shifted_up_green_channel_total = 0
gradient_shifted_down_green_channel_total = 0
gradient_shifted_left_green_channel_total = 0
gradient_shifted_right_green_channel_total = 0
gradient_shifted_up_blue_channel_total = 0
gradient_shifted_down_blue_channel_total = 0
gradient_shifted_left_blue_channel_total = 0
gradient_shifted_right_blue_channel_total = 0
gradient_shifted_up_hue_channel_total = 0
gradient_shifted_down_hue_channel_total = 0
gradient_shifted_left_hue_channel_total = 0
gradient_shifted_right_hue_channel_total = 0
gradient_shifted_up_saturation_channel_total = 0
gradient_shifted_down_saturation_channel_total = 0
gradient_shifted_left_saturation_channel_total = 0
gradient_shifted_right_saturation_channel_total = 0
gradient_shifted_up_value_channel_total = 0
gradient_shifted_down_value_channel_total = 0
gradient_shifted_left_value_channel_total = 0
gradient_shifted_right_value_channel_total = 0
print((cost-prevcost)/prevcost)
#loop until change in cost is less than 1 percent
while abs((prevcost-cost)/prevcost) > 0.01:
    prevcost = cost+0
    #update the weights
    #print(weights.reshape(1,24))
    weights = weights - learning_rate * np.array([gradient_shifted_up_red_channel_total, gradient_shifted_down_red_channel_total, gradient_shifted_left_red_channel_total, gradient_shifted_right_red_channel_total, gradient_shifted_up_green_channel_total, gradient_shifted_down_green_channel_total, gradient_shifted_left_green_channel_total, gradient_shifted_right_green_channel_total, gradient_shifted_up_blue_channel_total, gradient_shifted_down_blue_channel_total, gradient_shifted_left_blue_channel_total, gradient_shifted_right_blue_channel_total, gradient_shifted_up_hue_channel_total, gradient_shifted_down_hue_channel_total, gradient_shifted_left_hue_channel_total, gradient_shifted_right_hue_channel_total, gradient_shifted_up_saturation_channel_total, gradient_shifted_down_saturation_channel_total, gradient_shifted_left_saturation_channel_total, gradient_shifted_right_saturation_channel_total, gradient_shifted_up_value_channel_total, gradient_shifted_down_value_channel_total, gradient_shifted_left_value_channel_total, gradient_shifted_right_value_channel_total]).reshape(24,1)
    print(np.array([gradient_shifted_up_red_channel_total, gradient_shifted_down_red_channel_total, gradient_shifted_left_red_channel_total, gradient_shifted_right_red_channel_total, gradient_shifted_up_green_channel_total, gradient_shifted_down_green_channel_total, gradient_shifted_left_green_channel_total, gradient_shifted_right_green_channel_total, gradient_shifted_up_blue_channel_total, gradient_shifted_down_blue_channel_total, gradient_shifted_left_blue_channel_total, gradient_shifted_right_blue_channel_total, gradient_shifted_up_hue_channel_total, gradient_shifted_down_hue_channel_total, gradient_shifted_left_hue_channel_total, gradient_shifted_right_hue_channel_total, gradient_shifted_up_saturation_channel_total, gradient_shifted_down_saturation_channel_total, gradient_shifted_left_saturation_channel_total, gradient_shifted_right_saturation_channel_total, gradient_shifted_up_value_channel_total, gradient_shifted_down_value_channel_total, gradient_shifted_left_value_channel_total, gradient_shifted_right_value_channel_total]))
    #loop through all images in dataset
    gradient_shifted_up_red_channel_total = 0
    gradient_shifted_down_red_channel_total = 0
    gradient_shifted_left_red_channel_total = 0
    gradient_shifted_right_red_channel_total = 0
    gradient_shifted_up_green_channel_total = 0
    gradient_shifted_down_green_channel_total = 0
    gradient_shifted_left_green_channel_total = 0
    gradient_shifted_right_green_channel_total = 0
    gradient_shifted_up_blue_channel_total = 0
    gradient_shifted_down_blue_channel_total = 0
    gradient_shifted_left_blue_channel_total = 0
    gradient_shifted_right_blue_channel_total = 0
    gradient_shifted_up_hue_channel_total = 0
    gradient_shifted_down_hue_channel_total = 0
    gradient_shifted_left_hue_channel_total = 0
    gradient_shifted_right_hue_channel_total = 0
    gradient_shifted_up_saturation_channel_total = 0
    gradient_shifted_down_saturation_channel_total = 0
    gradient_shifted_left_saturation_channel_total = 0
    gradient_shifted_right_saturation_channel_total = 0
    gradient_shifted_up_value_channel_total = 0
    gradient_shifted_down_value_channel_total = 0
    gradient_shifted_left_value_channel_total = 0
    gradient_shifted_right_value_channel_total = 0
    cost = 0
    for i in range(0, numfiles):
        #load input image
        input_image = cv2.imread(dataset_folder + input_filenames[i])
        #load output image
        output_image = cv2.imread(dataset_folder + output_filenames[i])
        #get highest intensity pixel indices of output image
        ground_truth = get_highest_intensity_pixel_indices(output_image)
        #get red channel of input image
        red_channel = get_red_channnel(input_image)
        #get green channel of input image
        green_channel = get_green_channel(input_image)
        #get blue channel of input image
        blue_channel = get_blue_channel(input_image)
        #get hue channel of input image
        hue_channel = get_hue_channel(input_image)
        #get saturation channel of input image
        saturation_channel = get_saturation_channel(input_image)
        #get value channel of input image
        value_channel = get_value_channel(input_image)
        #get shifted up red channel of input image
        shifted_up_red_channel = shift_image_up(red_channel)
        #get shifted down red channel of input image
        shifted_down_red_channel = shift_image_down(red_channel)
        #get shifted left red channel of input image
        shifted_left_red_channel = shift_image_left(red_channel)
        #get shifted right red channel of input_image
        shifted_right_red_channel = shift_image_right(red_channel)
        #get shifted up green channel of input image
        shifted_up_green_channel = shift_image_up(green_channel)
        #get shifted down green channel of input image
        shifted_down_green_channel = shift_image_down(green_channel)
        #get shifted left green channel of input image
        shifted_left_green_channel = shift_image_left(green_channel)
        #get shifted right green channel of input image
        shifted_right_green_channel = shift_image_right(green_channel)
        #get shifted up blue channel of input image
        shifted_up_blue_channel = shift_image_up(blue_channel)
        #get shifted down blue channel of input image
        shifted_down_blue_channel = shift_image_down(blue_channel)
        #get shifted left blue channel of input image
        shifted_left_blue_channel = shift_image_left(blue_channel)
        #get shifted right blue channel of input image
        shifted_right_blue_channel = shift_image_right(blue_channel)
        #get shifted up hue channel of input image
        shifted_up_hue_channel = shift_image_up(hue_channel)
        #get shifted down hue channel of input image
        shifted_down_hue_channel = shift_image_down(hue_channel)
        #get shifted left hue channel
        shifted_left_hue_channel = shift_image_left(hue_channel)
        #get shifted right hue channel
        shifted_right_hue_channel = shift_image_right(hue_channel)
        #get shifted up saturation channel of input image
        shifted_up_saturation_channel = shift_image_up(saturation_channel)
        #get shifted down saturation channel of input image
        shifted_down_saturation_channel = shift_image_down(saturation_channel)
        #get shifted left saturation channel of input image
        shifted_left_saturation_channel = shift_image_left(saturation_channel)
        #get shifted right saturation channel of input image
        shifted_right_saturation_channel = shift_image_right(saturation_channel)
        #get shifted up value channel of input image
        shifted_up_value_channel = shift_image_up(value_channel)
        #get shifted down value channel of input image
        shifted_down_value_channel = shift_image_down(value_channel)
        #get shifted left value channel of input image
        shifted_left_value_channel = shift_image_left(value_channel)
        #get shifted right value channel of input image
        shifted_right_value_channel = shift_image_right(value_channel)

        shifted_up_red_channel_indices = get_highest_intensity_pixel_indices(shifted_up_red_channel)
        shifted_down_red_channel_indices = get_highest_intensity_pixel_indices(shifted_down_red_channel)
        shifted_left_red_channel_indices = get_highest_intensity_pixel_indices(shifted_left_red_channel)
        shifted_right_red_channel_indices = get_highest_intensity_pixel_indices(shifted_right_red_channel)
        shifted_up_green_channel_indices = get_highest_intensity_pixel_indices(shifted_up_green_channel)
        shifted_down_green_channel_indices = get_highest_intensity_pixel_indices(shifted_down_green_channel)
        shifted_left_green_channel_indices = get_highest_intensity_pixel_indices(shifted_left_green_channel)
        shifted_right_green_channel_indices = get_highest_intensity_pixel_indices(shifted_right_green_channel)
        shifted_up_blue_channel_indices = get_highest_intensity_pixel_indices(shifted_up_blue_channel)
        shifted_down_blue_channel_indices = get_highest_intensity_pixel_indices(shifted_down_blue_channel)
        shifted_left_blue_channel_indices = get_highest_intensity_pixel_indices(shifted_left_blue_channel)
        shifted_right_blue_channel_indices = get_highest_intensity_pixel_indices(shifted_right_blue_channel)
        shifted_up_hue_channel_indices = get_highest_intensity_pixel_indices(shifted_up_hue_channel)
        shifted_down_hue_channel_indices = get_highest_intensity_pixel_indices(shifted_down_hue_channel)
        shifted_left_hue_channel_indices = get_highest_intensity_pixel_indices(shifted_left_hue_channel)
        shifted_right_hue_channel_indices = get_highest_intensity_pixel_indices(shifted_right_hue_channel)
        shifted_up_saturation_channel_indices = get_highest_intensity_pixel_indices(shifted_up_saturation_channel)
        shifted_down_saturation_channel_indices = get_highest_intensity_pixel_indices(shifted_down_saturation_channel)
        shifted_left_saturation_channel_indices = get_highest_intensity_pixel_indices(shifted_left_saturation_channel)
        shifted_right_saturation_channel_indices = get_highest_intensity_pixel_indices(shifted_right_saturation_channel)
        shifted_up_value_channel_indices = get_highest_intensity_pixel_indices(shifted_up_value_channel)
        shifted_down_value_channel_indices = get_highest_intensity_pixel_indices(shifted_down_value_channel)
        shifted_left_value_channel_indices = get_highest_intensity_pixel_indices(shifted_left_value_channel)
        shifted_right_value_channel_indices = get_highest_intensity_pixel_indices(shifted_right_value_channel)

        weights_step_up_shifted_up_red_channel = weights+incr*np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_up_red_channel = weights-incr*np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_down_red_channel = weights+incr*np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_down_red_channel = weights-incr*np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_left_red_channel = weights+incr*np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_left_red_channel = weights-incr*np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_right_red_channel = weights+incr*np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_right_red_channel = weights-incr*np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_up_green_channel = weights+incr*np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_up_green_channel = weights-incr*np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_down_green_channel = weights+incr*np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_down_green_channel = weights-incr*np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_left_green_channel = weights+incr*np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_left_green_channel = weights-incr*np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_right_green_channel = weights+incr*np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_right_green_channel = weights-incr*np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_up_blue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_up_blue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_down_blue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_down_blue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_left_blue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_left_blue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_right_blue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_right_blue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_up_hue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_up_hue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_down_hue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_down_hue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_left_hue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_left_hue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_right_hue_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        weights_step_down_shifted_right_hue_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        weights_step_up_shifted_up_saturation_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        weights_step_down_shifted_up_saturation_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        weights_step_up_shifted_down_saturation_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        weights_step_down_shifted_down_saturation_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        weights_step_up_shifted_left_saturation_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        weights_step_down_shifted_left_saturation_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        weights_step_up_shifted_right_saturation_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        weights_step_down_shifted_right_saturation_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        weights_step_up_shifted_up_value_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        weights_step_down_shifted_up_value_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        weights_step_up_shifted_down_value_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        weights_step_down_shifted_down_value_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        weights_step_up_shifted_left_value_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        weights_step_down_shifted_left_value_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        weights_step_up_shifted_right_value_channel = weights+incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        weights_step_down_shifted_right_value_channel = weights-incr*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

        weighted_average = get_weighted_average(weights, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)

        weighted_average_step_up_shifted_up_red_channel = get_weighted_average(weights_step_up_shifted_up_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_up_red_channel = get_weighted_average(weights_step_down_shifted_up_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_down_red_channel = get_weighted_average(weights_step_up_shifted_down_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_down_red_channel = get_weighted_average(weights_step_down_shifted_down_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_left_red_channel = get_weighted_average(weights_step_up_shifted_left_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_left_red_channel = get_weighted_average(weights_step_down_shifted_left_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_right_red_channel = get_weighted_average(weights_step_up_shifted_right_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_right_red_channel = get_weighted_average(weights_step_down_shifted_right_red_channel, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_up_green_channel = get_weighted_average(weights_step_up_shifted_up_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_up_green_channel = get_weighted_average(weights_step_down_shifted_up_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_down_green_channel = get_weighted_average(weights_step_up_shifted_down_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_down_green_channel = get_weighted_average(weights_step_down_shifted_down_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_left_green_channel = get_weighted_average(weights_step_up_shifted_left_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_left_green_channel = get_weighted_average(weights_step_down_shifted_left_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_right_green_channel = get_weighted_average(weights_step_up_shifted_right_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_right_green_channel = get_weighted_average(weights_step_down_shifted_right_green_channel, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_up_blue_channel = get_weighted_average(weights_step_up_shifted_up_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_up_blue_channel = get_weighted_average(weights_step_down_shifted_up_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_down_blue_channel = get_weighted_average(weights_step_up_shifted_down_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_down_blue_channel = get_weighted_average(weights_step_down_shifted_down_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_left_blue_channel = get_weighted_average(weights_step_up_shifted_left_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_left_blue_channel = get_weighted_average(weights_step_down_shifted_left_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_right_blue_channel = get_weighted_average(weights_step_up_shifted_right_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_right_blue_channel = get_weighted_average(weights_step_down_shifted_right_blue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_up_hue_channel = get_weighted_average(weights_step_up_shifted_up_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_up_hue_channel = get_weighted_average(weights_step_down_shifted_up_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_down_hue_channel = get_weighted_average(weights_step_up_shifted_down_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_down_hue_channel = get_weighted_average(weights_step_down_shifted_down_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_left_hue_channel = get_weighted_average(weights_step_up_shifted_left_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_left_hue_channel = get_weighted_average(weights_step_down_shifted_left_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_right_hue_channel = get_weighted_average(weights_step_up_shifted_right_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_right_hue_channel = get_weighted_average(weights_step_down_shifted_right_hue_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_up_saturation_channel = get_weighted_average(weights_step_up_shifted_up_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_up_saturation_channel = get_weighted_average(weights_step_down_shifted_up_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_down_saturation_channel = get_weighted_average(weights_step_up_shifted_down_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_down_saturation_channel = get_weighted_average(weights_step_down_shifted_down_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_left_saturation_channel = get_weighted_average(weights_step_up_shifted_left_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_left_saturation_channel = get_weighted_average(weights_step_down_shifted_left_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_right_saturation_channel = get_weighted_average(weights_step_up_shifted_right_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_right_saturation_channel = get_weighted_average(weights_step_down_shifted_right_saturation_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_up_value_channel = get_weighted_average(weights_step_up_shifted_up_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_up_value_channel = get_weighted_average(weights_step_down_shifted_up_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_down_value_channel = get_weighted_average(weights_step_up_shifted_down_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_down_value_channel = get_weighted_average(weights_step_down_shifted_down_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_left_value_channel = get_weighted_average(weights_step_up_shifted_left_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_left_value_channel = get_weighted_average(weights_step_down_shifted_left_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_up_shifted_right_value_channel = get_weighted_average(weights_step_up_shifted_right_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)
        weighted_average_step_down_shifted_right_value_channel = get_weighted_average(weights_step_down_shifted_right_value_channel, shifted_up_blue_channel_indices, shifted_down_blue_channel_indices, shifted_left_blue_channel_indices, shifted_right_blue_channel_indices, shifted_up_red_channel_indices, shifted_down_red_channel_indices, shifted_left_red_channel_indices, shifted_right_red_channel_indices, shifted_up_green_channel_indices, shifted_down_green_channel_indices, shifted_left_green_channel_indices, shifted_right_green_channel_indices, shifted_up_hue_channel_indices, shifted_down_hue_channel_indices, shifted_left_hue_channel_indices, shifted_right_hue_channel_indices, shifted_up_saturation_channel_indices, shifted_down_saturation_channel_indices, shifted_left_saturation_channel_indices, shifted_right_saturation_channel_indices, shifted_up_value_channel_indices, shifted_down_value_channel_indices, shifted_left_value_channel_indices, shifted_right_value_channel_indices)

        #compute costs from all 48 weighted averages
        cost_step_up_shifted_up_red_channel = compute_cost(weighted_average_step_up_shifted_up_red_channel, ground_truth)
        cost_step_down_shifted_up_red_channel = compute_cost(weighted_average_step_down_shifted_up_red_channel, ground_truth)
        cost_step_up_shifted_down_red_channel = compute_cost(weighted_average_step_up_shifted_down_red_channel, ground_truth)
        cost_step_down_shifted_down_red_channel = compute_cost(weighted_average_step_down_shifted_down_red_channel, ground_truth)
        cost_step_up_shifted_left_red_channel = compute_cost(weighted_average_step_up_shifted_left_red_channel, ground_truth)
        cost_step_down_shifted_left_red_channel = compute_cost(weighted_average_step_down_shifted_left_red_channel, ground_truth)
        cost_step_up_shifted_right_red_channel = compute_cost(weighted_average_step_up_shifted_right_red_channel, ground_truth)
        cost_step_down_shifted_right_red_channel = compute_cost(weighted_average_step_down_shifted_right_red_channel, ground_truth)
        cost_step_up_shifted_up_green_channel = compute_cost(weighted_average_step_up_shifted_up_green_channel, ground_truth)
        cost_step_down_shifted_up_green_channel = compute_cost(weighted_average_step_down_shifted_up_green_channel, ground_truth)
        cost_step_up_shifted_down_green_channel = compute_cost(weighted_average_step_up_shifted_down_green_channel, ground_truth)
        cost_step_down_shifted_down_green_channel = compute_cost(weighted_average_step_down_shifted_down_green_channel, ground_truth)
        cost_step_up_shifted_left_green_channel = compute_cost(weighted_average_step_up_shifted_left_green_channel, ground_truth)
        cost_step_down_shifted_left_green_channel = compute_cost(weighted_average_step_down_shifted_left_green_channel, ground_truth)
        cost_step_up_shifted_right_green_channel = compute_cost(weighted_average_step_up_shifted_right_green_channel, ground_truth)
        cost_step_down_shifted_right_green_channel = compute_cost(weighted_average_step_down_shifted_right_green_channel, ground_truth)
        cost_step_up_shifted_up_blue_channel = compute_cost(weighted_average_step_up_shifted_up_blue_channel, ground_truth)
        cost_step_down_shifted_up_blue_channel = compute_cost(weighted_average_step_down_shifted_up_blue_channel, ground_truth)
        cost_step_up_shifted_down_blue_channel = compute_cost(weighted_average_step_up_shifted_down_blue_channel, ground_truth)
        cost_step_down_shifted_down_blue_channel = compute_cost(weighted_average_step_down_shifted_down_blue_channel, ground_truth)
        cost_step_up_shifted_left_blue_channel = compute_cost(weighted_average_step_up_shifted_left_blue_channel, ground_truth)
        cost_step_down_shifted_left_blue_channel = compute_cost(weighted_average_step_down_shifted_left_blue_channel, ground_truth)
        cost_step_up_shifted_right_blue_channel = compute_cost(weighted_average_step_up_shifted_right_blue_channel, ground_truth)
        cost_step_down_shifted_right_blue_channel = compute_cost(weighted_average_step_down_shifted_right_blue_channel, ground_truth)
        cost_step_up_shifted_up_hue_channel = compute_cost(weighted_average_step_up_shifted_up_hue_channel, ground_truth)
        cost_step_down_shifted_up_hue_channel = compute_cost(weighted_average_step_down_shifted_up_hue_channel, ground_truth)
        cost_step_up_shifted_down_hue_channel = compute_cost(weighted_average_step_up_shifted_down_hue_channel, ground_truth)
        cost_step_down_shifted_down_hue_channel = compute_cost(weighted_average_step_down_shifted_down_hue_channel, ground_truth)
        cost_step_up_shifted_left_hue_channel = compute_cost(weighted_average_step_up_shifted_left_hue_channel, ground_truth)
        cost_step_down_shifted_left_hue_channel = compute_cost(weighted_average_step_down_shifted_left_hue_channel, ground_truth)
        cost_step_up_shifted_right_hue_channel = compute_cost(weighted_average_step_up_shifted_right_hue_channel, ground_truth)
        cost_step_down_shifted_right_hue_channel = compute_cost(weighted_average_step_down_shifted_right_hue_channel, ground_truth)
        cost_step_up_shifted_up_saturation_channel = compute_cost(weighted_average_step_up_shifted_up_saturation_channel, ground_truth)
        cost_step_down_shifted_up_saturation_channel = compute_cost(weighted_average_step_down_shifted_up_saturation_channel, ground_truth)
        cost_step_up_shifted_down_saturation_channel = compute_cost(weighted_average_step_up_shifted_down_saturation_channel, ground_truth)
        cost_step_down_shifted_down_saturation_channel = compute_cost(weighted_average_step_down_shifted_down_saturation_channel, ground_truth)
        cost_step_up_shifted_left_saturation_channel = compute_cost(weighted_average_step_up_shifted_left_saturation_channel, ground_truth)
        cost_step_down_shifted_left_saturation_channel = compute_cost(weighted_average_step_down_shifted_left_saturation_channel, ground_truth)
        cost_step_up_shifted_right_saturation_channel = compute_cost(weighted_average_step_up_shifted_right_saturation_channel, ground_truth)
        cost_step_down_shifted_right_saturation_channel = compute_cost(weighted_average_step_down_shifted_right_saturation_channel, ground_truth)
        cost_step_up_shifted_up_value_channel = compute_cost(weighted_average_step_up_shifted_up_value_channel, ground_truth)
        cost_step_down_shifted_up_value_channel = compute_cost(weighted_average_step_down_shifted_up_value_channel, ground_truth)
        cost_step_up_shifted_down_value_channel = compute_cost(weighted_average_step_up_shifted_down_value_channel, ground_truth)
        cost_step_down_shifted_down_value_channel = compute_cost(weighted_average_step_down_shifted_down_value_channel, ground_truth)
        cost_step_up_shifted_left_value_channel = compute_cost(weighted_average_step_up_shifted_left_value_channel, ground_truth)
        cost_step_down_shifted_left_value_channel = compute_cost(weighted_average_step_down_shifted_left_value_channel, ground_truth)
        cost_step_up_shifted_right_value_channel = compute_cost(weighted_average_step_up_shifted_right_value_channel, ground_truth)
        cost_step_down_shifted_right_value_channel = compute_cost(weighted_average_step_down_shifted_right_value_channel, ground_truth)

        #Compute the 24 gradients of the cost
        gradient_shifted_up_red_channel = (cost_step_up_shifted_up_red_channel-cost_step_down_shifted_up_red_channel)/(2*incr)
        gradient_shifted_down_red_channel = (cost_step_up_shifted_down_red_channel-cost_step_down_shifted_down_red_channel)/(2*incr)
        gradient_shifted_left_red_channel = (cost_step_up_shifted_left_red_channel-cost_step_down_shifted_left_red_channel)/(2*incr)
        gradient_shifted_right_red_channel = (cost_step_up_shifted_right_red_channel-cost_step_down_shifted_right_red_channel)/(2*incr)
        gradient_shifted_up_green_channel = (cost_step_up_shifted_up_green_channel-cost_step_down_shifted_up_green_channel)/(2*incr)
        gradient_shifted_down_green_channel = (cost_step_up_shifted_down_green_channel-cost_step_down_shifted_down_green_channel)/(2*incr)
        gradient_shifted_left_green_channel = (cost_step_up_shifted_left_green_channel-cost_step_down_shifted_left_green_channel)/(2*incr)
        gradient_shifted_right_green_channel = (cost_step_up_shifted_right_green_channel-cost_step_down_shifted_right_green_channel)/(2*incr)
        gradient_shifted_up_blue_channel = (cost_step_up_shifted_up_blue_channel-cost_step_down_shifted_up_blue_channel)/(2*incr)
        gradient_shifted_down_blue_channel = (cost_step_up_shifted_down_blue_channel-cost_step_down_shifted_down_blue_channel)/(2*incr)
        gradient_shifted_left_blue_channel = (cost_step_up_shifted_left_blue_channel-cost_step_down_shifted_left_blue_channel)/(2*incr)
        gradient_shifted_right_blue_channel = (cost_step_up_shifted_right_blue_channel-cost_step_down_shifted_right_blue_channel)/(2*incr)
        gradient_shifted_up_hue_channel = (cost_step_up_shifted_up_hue_channel-cost_step_down_shifted_up_hue_channel)/(2*incr)
        gradient_shifted_down_hue_channel = (cost_step_up_shifted_down_hue_channel-cost_step_down_shifted_down_hue_channel)/(2*incr)
        gradient_shifted_left_hue_channel = (cost_step_up_shifted_left_hue_channel-cost_step_down_shifted_left_hue_channel)/(2*incr)
        gradient_shifted_right_hue_channel = (cost_step_up_shifted_right_hue_channel-cost_step_down_shifted_right_hue_channel)/(2*incr)
        gradient_shifted_up_saturation_channel = (cost_step_up_shifted_up_saturation_channel-cost_step_down_shifted_up_saturation_channel)/(2*incr)
        gradient_shifted_down_saturation_channel = (cost_step_up_shifted_down_saturation_channel-cost_step_down_shifted_down_saturation_channel)/(2*incr)
        gradient_shifted_left_saturation_channel = (cost_step_up_shifted_left_saturation_channel-cost_step_down_shifted_left_saturation_channel)/(2*incr)
        gradient_shifted_right_saturation_channel = (cost_step_up_shifted_right_saturation_channel-cost_step_down_shifted_right_saturation_channel)/(2*incr)
        gradient_shifted_up_value_channel = (cost_step_up_shifted_up_value_channel-cost_step_down_shifted_up_value_channel)/(2*incr)
        gradient_shifted_down_value_channel = (cost_step_up_shifted_down_value_channel-cost_step_down_shifted_down_value_channel)/(2*incr)
        gradient_shifted_left_value_channel = (cost_step_up_shifted_left_value_channel-cost_step_down_shifted_left_value_channel)/(2*incr)
        gradient_shifted_right_value_channel = (cost_step_up_shifted_right_value_channel-cost_step_down_shifted_right_value_channel)/(2*incr)
        
        #Compute the total gradient
        gradient_shifted_up_red_channel_total = gradient_shifted_up_red_channel_total+gradient_shifted_up_red_channel
        gradient_shifted_down_red_channel_total = gradient_shifted_down_red_channel_total+gradient_shifted_down_red_channel
        gradient_shifted_left_red_channel_total = gradient_shifted_left_red_channel_total+gradient_shifted_left_red_channel
        gradient_shifted_right_red_channel_total = gradient_shifted_right_red_channel_total+gradient_shifted_right_red_channel
        gradient_shifted_up_green_channel_total = gradient_shifted_up_green_channel_total+gradient_shifted_up_green_channel
        gradient_shifted_down_green_channel_total = gradient_shifted_down_green_channel_total+gradient_shifted_down_green_channel
        gradient_shifted_left_green_channel_total = gradient_shifted_left_green_channel_total+gradient_shifted_left_green_channel
        gradient_shifted_right_green_channel_total = gradient_shifted_right_green_channel_total+gradient_shifted_right_green_channel
        gradient_shifted_up_blue_channel_total = gradient_shifted_up_blue_channel_total+gradient_shifted_up_blue_channel
        gradient_shifted_down_blue_channel_total = gradient_shifted_down_blue_channel_total+gradient_shifted_down_blue_channel
        gradient_shifted_left_blue_channel_total = gradient_shifted_left_blue_channel_total+gradient_shifted_left_blue_channel
        gradient_shifted_right_blue_channel_total = gradient_shifted_right_blue_channel_total+gradient_shifted_right_blue_channel
        gradient_shifted_up_hue_channel_total = gradient_shifted_up_hue_channel_total+gradient_shifted_up_hue_channel
        gradient_shifted_down_hue_channel_total = gradient_shifted_down_hue_channel_total+gradient_shifted_down_hue_channel
        gradient_shifted_left_hue_channel_total = gradient_shifted_left_hue_channel_total+gradient_shifted_left_hue_channel
        gradient_shifted_right_hue_channel_total = gradient_shifted_right_hue_channel_total+gradient_shifted_right_hue_channel
        gradient_shifted_up_saturation_channel_total = gradient_shifted_up_saturation_channel_total+gradient_shifted_up_saturation_channel
        gradient_shifted_down_saturation_channel_total = gradient_shifted_down_saturation_channel_total+gradient_shifted_down_saturation_channel
        gradient_shifted_left_saturation_channel_total = gradient_shifted_left_saturation_channel_total+gradient_shifted_left_saturation_channel
        gradient_shifted_right_saturation_channel_total = gradient_shifted_right_saturation_channel_total+gradient_shifted_right_saturation_channel
        gradient_shifted_up_value_channel_total = gradient_shifted_up_value_channel_total+gradient_shifted_up_value_channel
        gradient_shifted_down_value_channel_total = gradient_shifted_down_value_channel_total+gradient_shifted_down_value_channel
        gradient_shifted_left_value_channel_total = gradient_shifted_left_value_channel_total+gradient_shifted_left_value_channel
        gradient_shifted_right_value_channel_total = gradient_shifted_right_value_channel_total+gradient_shifted_right_value_channel

        cost = cost+compute_cost(weighted_average, ground_truth)