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
    cost = np.sum(np.square(array1 - array2))
    return cost

def get_highest_intensity_pixel_indices(image):
    highest_intensity_pixel_indices = np.zeros((1,image.shape[1])).astype(int)
    for i in range(image.shape[1]):
        highest_intensity_pixel_indices[0,i] = np.argmax(image[:,i])
    return highest_intensity_pixel_indices

def get_red_channel(image):
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

#load output images as grayscale and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_output_indices(output_filenames):
    output_indices = np.zeros((0,0)).astype(int)
    numfiles = len(output_filenames)
    for i in range(0, numfiles):
        output_image = cv2.imread(dataset_folder + output_filenames[i], cv2.IMREAD_GRAYSCALE)
        output_indices = np.append(output_indices, get_highest_intensity_pixel_indices(output_image))
    return output_indices

#load input images as red channel and shift upward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_red_shift_up_indices(input_filenames):
    input_red_shift_up_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_red_channel = get_red_channel(input_image)
        input_red_shift_up_channel = shift_image_up(input_red_channel)
        input_red_shift_up_indices = np.append(input_red_shift_up_indices, get_highest_intensity_pixel_indices(input_red_shift_up_channel))
    return input_red_shift_up_indices

#load input images as red channel and shift downward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_red_shift_down_indices(input_filenames):
    input_red_shift_down_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_red_channel = get_red_channel(input_image)
        input_red_shift_down_channel = shift_image_down(input_red_channel)
        input_red_shift_down_indices = np.append(input_red_shift_down_indices, get_highest_intensity_pixel_indices(input_red_shift_down_channel))
    return input_red_shift_down_indices

#load input images as red channel and shift left and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_red_shift_left_indices(input_filenames):
    input_red_shift_left_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_red_channel = get_red_channel(input_image)
        input_red_shift_left_channel = shift_image_left(input_red_channel)
        input_red_shift_left_indices = np.append(input_red_shift_left_indices, get_highest_intensity_pixel_indices(input_red_shift_left_channel))
    return input_red_shift_left_indices

#load input images as red channel and shift right and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_red_shift_right_indices(input_filenames):
    input_red_shift_right_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_red_channel = get_red_channel(input_image)
        input_red_shift_right_channel = shift_image_right(input_red_channel)
        input_red_shift_right_indices = np.append(input_red_shift_right_indices, get_highest_intensity_pixel_indices(input_red_shift_right_channel))
    return input_red_shift_right_indices

#load input images as green channel and shift upward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_green_shift_up_indices(input_filenames):
    input_green_shift_up_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_green_channel = get_green_channel(input_image)
        input_green_shift_up_channel = shift_image_up(input_green_channel)
        input_green_shift_up_indices = np.append(input_green_shift_up_indices, get_highest_intensity_pixel_indices(input_green_shift_up_channel))
    return input_green_shift_up_indices

#load input images as green channel and shift downward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_green_shift_down_indices(input_filenames):
    input_green_shift_down_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_green_channel = get_green_channel(input_image)
        input_green_shift_down_channel = shift_image_down(input_green_channel)
        input_green_shift_down_indices = np.append(input_green_shift_down_indices, get_highest_intensity_pixel_indices(input_green_shift_down_channel))
    return input_green_shift_down_indices

#load input images as green channel and shift left and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_green_shift_left_indices(input_filenames):
    input_green_shift_left_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_green_channel = get_green_channel(input_image)
        input_green_shift_left_channel = shift_image_left(input_green_channel)
        input_green_shift_left_indices = np.append(input_green_shift_left_indices, get_highest_intensity_pixel_indices(input_green_shift_left_channel))
    return input_green_shift_left_indices

#load input images as green channel and shift right and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_green_shift_right_indices(input_filenames):
    input_green_shift_right_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_green_channel = get_green_channel(input_image)
        input_green_shift_right_channel = shift_image_right(input_green_channel)
        input_green_shift_right_indices = np.append(input_green_shift_right_indices, get_highest_intensity_pixel_indices(input_green_shift_right_channel))
    return input_green_shift_right_indices

#load input images as blue channel and shift upward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_blue_shift_up_indices(input_filenames):
    input_blue_shift_up_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_blue_channel = get_blue_channel(input_image)
        input_blue_shift_up_channel = shift_image_up(input_blue_channel)
        input_blue_shift_up_indices = np.append(input_blue_shift_up_indices, get_highest_intensity_pixel_indices(input_blue_shift_up_channel))
    return input_blue_shift_up_indices

#load input images as blue channel and shift downward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_blue_shift_down_indices(input_filenames):
    input_blue_shift_down_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_blue_channel = get_blue_channel(input_image)
        input_blue_shift_down_channel = shift_image_down(input_blue_channel)
        input_blue_shift_down_indices = np.append(input_blue_shift_down_indices, get_highest_intensity_pixel_indices(input_blue_shift_down_channel))
    return input_blue_shift_down_indices

#load input images as blue channel and shift left and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_blue_shift_left_indices(input_filenames):
    input_blue_shift_left_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_blue_channel = get_blue_channel(input_image)
        input_blue_shift_left_channel = shift_image_left(input_blue_channel)
        input_blue_shift_left_indices = np.append(input_blue_shift_left_indices, get_highest_intensity_pixel_indices(input_blue_shift_left_channel))
    return input_blue_shift_left_indices

#load input images as blue channel and shift right and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_blue_shift_right_indices(input_filenames):
    input_blue_shift_right_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_blue_channel = get_blue_channel(input_image)
        input_blue_shift_right_channel = shift_image_right(input_blue_channel)
        input_blue_shift_right_indices = np.append(input_blue_shift_right_indices, get_highest_intensity_pixel_indices(input_blue_shift_right_channel))
    return input_blue_shift_right_indices

#load input images as hue channel and shift upward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_hue_shift_up_indices(input_filenames):
    input_hue_shift_up_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_hue_channel = get_hue_channel(input_image)
        input_hue_shift_up_channel = shift_image_up(input_hue_channel)
        input_hue_shift_up_indices = np.append(input_hue_shift_up_indices, get_highest_intensity_pixel_indices(input_hue_shift_up_channel))
    return input_hue_shift_up_indices

#load input images as hue channel and shift downward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_hue_shift_down_indices(input_filenames):
    input_hue_shift_down_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_hue_channel = get_hue_channel(input_image)
        input_hue_shift_down_channel = shift_image_down(input_hue_channel)
        input_hue_shift_down_indices = np.append(input_hue_shift_down_indices, get_highest_intensity_pixel_indices(input_hue_shift_down_channel))
    return input_hue_shift_down_indices

#load input images as hue channel and shift left and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_hue_shift_left_indices(input_filenames):
    input_hue_shift_left_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_hue_channel = get_hue_channel(input_image)
        input_hue_shift_left_channel = shift_image_left(input_hue_channel)
        input_hue_shift_left_indices = np.append(input_hue_shift_left_indices, get_highest_intensity_pixel_indices(input_hue_shift_left_channel))
    return input_hue_shift_left_indices

#load input images as hue channel and shift right and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_hue_shift_right_indices(input_filenames):
    input_hue_shift_right_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_hue_channel = get_hue_channel(input_image)
        input_hue_shift_right_channel = shift_image_right(input_hue_channel)
        input_hue_shift_right_indices = np.append(input_hue_shift_right_indices, get_highest_intensity_pixel_indices(input_hue_shift_right_channel))
    return input_hue_shift_right_indices

#load input images as saturation channel and shift upward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_saturation_shift_up_indices(input_filenames):
    input_saturation_shift_up_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_saturation_channel = get_saturation_channel(input_image)
        input_saturation_shift_up_channel = shift_image_up(input_saturation_channel)
        input_saturation_shift_up_indices = np.append(input_saturation_shift_up_indices, get_highest_intensity_pixel_indices(input_saturation_shift_up_channel))
    return input_saturation_shift_up_indices

#load input images as saturation channel and shift downward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_saturation_shift_down_indices(input_filenames):
    input_saturation_shift_down_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_saturation_channel = get_saturation_channel(input_image)
        input_saturation_shift_down_channel = shift_image_down(input_saturation_channel)
        input_saturation_shift_down_indices = np.append(input_saturation_shift_down_indices, get_highest_intensity_pixel_indices(input_saturation_shift_down_channel))
    return input_saturation_shift_down_indices

#load input images as saturation channel and shift left and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_saturation_shift_left_indices(input_filenames):
    input_saturation_shift_left_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_saturation_channel = get_saturation_channel(input_image)
        input_saturation_shift_left_channel = shift_image_left(input_saturation_channel)
        input_saturation_shift_left_indices = np.append(input_saturation_shift_left_indices, get_highest_intensity_pixel_indices(input_saturation_shift_left_channel))
    return input_saturation_shift_left_indices

#load input images as saturation channel and shift right and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_saturation_shift_right_indices(input_filenames):
    input_saturation_shift_right_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_saturation_channel = get_saturation_channel(input_image)
        input_saturation_shift_right_channel = shift_image_right(input_saturation_channel)
        input_saturation_shift_right_indices = np.append(input_saturation_shift_right_indices, get_highest_intensity_pixel_indices(input_saturation_shift_right_channel))
    return input_saturation_shift_right_indices

#load input images as value channel and shift upward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_value_shift_up_indices(input_filenames):
    input_value_shift_up_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_value_channel = get_value_channel(input_image)
        input_value_shift_up_channel = shift_image_up(input_value_channel)
        input_value_shift_up_indices = np.append(input_value_shift_up_indices, get_highest_intensity_pixel_indices(input_value_shift_up_channel))
    return input_value_shift_up_indices

#load input images as value channel and shift downward and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_value_shift_down_indices(input_filenames):
    input_value_shift_down_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_value_channel = get_value_channel(input_image)
        input_value_shift_down_channel = shift_image_down(input_value_channel)
        input_value_shift_down_indices = np.append(input_value_shift_down_indices, get_highest_intensity_pixel_indices(input_value_shift_down_channel))
    return input_value_shift_down_indices

#load input images as value channel and shift left and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_value_shift_left_indices(input_filenames):
    input_value_shift_left_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_value_channel = get_value_channel(input_image)
        input_value_shift_left_channel = shift_image_left(input_value_channel)
        input_value_shift_left_indices = np.append(input_value_shift_left_indices, get_highest_intensity_pixel_indices(input_value_shift_left_channel))
    return input_value_shift_left_indices

#load input images as value channel and shift right and get vertical index of highest intensity pixel in each column and put all indices in one array
def get_input_value_shift_right_indices(input_filenames):
    input_value_shift_right_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_value_channel = get_value_channel(input_image)
        input_value_shift_right_channel = shift_image_right(input_value_channel)
        input_value_shift_right_indices = np.append(input_value_shift_right_indices, get_highest_intensity_pixel_indices(input_value_shift_right_channel))
    return input_value_shift_right_indices

#given increment and index, alter one element of array by increment
def alter_one_element(weights, increment, index):
    weights[index] = weights[index] + increment
    return weights

def gradient_descent(weights, input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices, output_indices, learning_rate, incr):
    weighted_average = get_weighted_average(weights, input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices)
    #find gradient of the cost for each set of indices
    orig_cost = compute_cost(weighted_average, output_indices)
    input_red_shift_up_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 0), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_red_shift_down_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 1), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_red_shift_left_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 2), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_red_shift_right_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 3), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_green_shift_up_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 4), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_green_shift_down_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 5), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_green_shift_left_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 6), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_green_shift_right_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 7), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_blue_shift_up_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 8), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_blue_shift_down_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 9), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_blue_shift_left_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 10), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_blue_shift_right_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 11), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_hue_shift_up_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 12), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_hue_shift_down_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 13), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_hue_shift_left_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 14), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_hue_shift_right_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 15), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_saturation_shift_up_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 16), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_saturation_shift_down_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 17), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_saturation_shift_left_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 18), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_saturation_shift_right_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 19), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_value_shift_up_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 20), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_value_shift_down_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 21), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_value_shift_left_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 22), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    input_value_shift_right_indices_cost = compute_cost(get_weighted_average(alter_one_element(weights, incr, 23), input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices), output_indices)
    cost_array = np.array([input_red_shift_up_indices_cost, input_red_shift_down_indices_cost, input_red_shift_left_indices_cost, input_red_shift_right_indices_cost, input_green_shift_up_indices_cost, input_green_shift_down_indices_cost, input_green_shift_left_indices_cost, input_green_shift_right_indices_cost, input_blue_shift_up_indices_cost, input_blue_shift_down_indices_cost, input_blue_shift_left_indices_cost, input_blue_shift_right_indices_cost, input_hue_shift_up_indices_cost, input_hue_shift_down_indices_cost, input_hue_shift_left_indices_cost, input_hue_shift_right_indices_cost, input_saturation_shift_up_indices_cost, input_saturation_shift_down_indices_cost, input_saturation_shift_left_indices_cost, input_saturation_shift_right_indices_cost, input_value_shift_up_indices_cost, input_value_shift_down_indices_cost, input_value_shift_left_indices_cost, input_value_shift_right_indices_cost])
    gradient_array = (cost_array-orig_cost)/incr
    #print(cost_array)
    #input("Press Enter to continue...")
    #print(gradient_array)
    #input("Press Enter to continue...")
    new_weights = weights - (learning_rate * incr * gradient_array.reshape(24, 1))
    #print(new_weights)
    #input("Press Enter to continue...")
    return new_weights


output_indices = get_output_indices(output_filenames)
input_red_shift_up_indices = get_input_red_shift_up_indices(input_filenames)
input_red_shift_down_indices = get_input_red_shift_down_indices(input_filenames)
input_red_shift_left_indices = get_input_red_shift_left_indices(input_filenames)
input_red_shift_right_indices = get_input_red_shift_right_indices(input_filenames)
input_green_shift_up_indices = get_input_green_shift_up_indices(input_filenames)
input_green_shift_down_indices = get_input_green_shift_down_indices(input_filenames)
input_green_shift_left_indices = get_input_green_shift_left_indices(input_filenames)
input_green_shift_right_indices = get_input_green_shift_right_indices(input_filenames)
input_blue_shift_up_indices = get_input_blue_shift_up_indices(input_filenames)
input_blue_shift_down_indices = get_input_blue_shift_down_indices(input_filenames)
input_blue_shift_left_indices = get_input_blue_shift_left_indices(input_filenames)
input_blue_shift_right_indices = get_input_blue_shift_right_indices(input_filenames)
input_hue_shift_up_indices = get_input_hue_shift_up_indices(input_filenames)
input_hue_shift_down_indices = get_input_hue_shift_down_indices(input_filenames)
input_hue_shift_left_indices = get_input_hue_shift_left_indices(input_filenames)
input_hue_shift_right_indices = get_input_hue_shift_right_indices(input_filenames)
input_saturation_shift_up_indices = get_input_saturation_shift_up_indices(input_filenames)
input_saturation_shift_down_indices = get_input_saturation_shift_down_indices(input_filenames)
input_saturation_shift_left_indices = get_input_saturation_shift_left_indices(input_filenames)
input_saturation_shift_right_indices = get_input_saturation_shift_right_indices(input_filenames)
input_value_shift_up_indices = get_input_value_shift_up_indices(input_filenames)
input_value_shift_down_indices = get_input_value_shift_down_indices(input_filenames)
input_value_shift_left_indices = get_input_value_shift_left_indices(input_filenames)
input_value_shift_right_indices = get_input_value_shift_right_indices(input_filenames)

#create 24x1 numpy array of ones
weights = np.ones((24,1))*0
weights[0] = 1
weights[1] = 1
weights[2] = 1
weights[3] = 0

#weighted average of all indices
weighted_average = get_weighted_average(weights, input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices)
cost = compute_cost(weighted_average, output_indices)
print(cost)
prevcost = 2*cost

incr = 0.01

learning_rate = 0.0000001

#gradient descent until cost changes by less than 1 percent
while abs(cost-prevcost) > 0.0001*prevcost:
    print(cost)
    prevcost = cost
    weights = gradient_descent(weights, input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices, output_indices, learning_rate, incr)
    weighted_average = get_weighted_average(weights, input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices)
    cost = compute_cost(weighted_average, output_indices)

print(weights.reshape(24,1))

#test on test images
for i in range(len(input_filenames)):
    #make a list containing only the ith element of input_filenames
    input_filename = [input_filenames[i]]
    input_red_shift_up_indices = get_input_red_shift_up_indices(input_filename)
    input_red_shift_down_indices = get_input_red_shift_down_indices(input_filename)
    input_red_shift_left_indices = get_input_red_shift_left_indices(input_filename)
    input_red_shift_right_indices = get_input_red_shift_right_indices(input_filename)
    input_green_shift_up_indices = get_input_green_shift_up_indices(input_filename)
    input_green_shift_down_indices = get_input_green_shift_down_indices(input_filename)
    input_green_shift_left_indices = get_input_green_shift_left_indices(input_filename)
    input_green_shift_right_indices = get_input_green_shift_right_indices(input_filename)
    input_blue_shift_up_indices = get_input_blue_shift_up_indices(input_filename)
    input_blue_shift_down_indices = get_input_blue_shift_down_indices(input_filename)
    input_blue_shift_left_indices = get_input_blue_shift_left_indices(input_filename)
    input_blue_shift_right_indices = get_input_blue_shift_right_indices(input_filename)
    input_hue_shift_up_indices = get_input_hue_shift_up_indices(input_filename)
    input_hue_shift_down_indices = get_input_hue_shift_down_indices(input_filename)
    input_hue_shift_left_indices = get_input_hue_shift_left_indices(input_filename)
    input_hue_shift_right_indices = get_input_hue_shift_right_indices(input_filename)
    input_saturation_shift_up_indices = get_input_saturation_shift_up_indices(input_filename)
    input_saturation_shift_down_indices = get_input_saturation_shift_down_indices(input_filename)
    input_saturation_shift_left_indices = get_input_saturation_shift_left_indices(input_filename)
    input_saturation_shift_right_indices = get_input_saturation_shift_right_indices(input_filename)
    input_value_shift_up_indices = get_input_value_shift_up_indices(input_filename)
    input_value_shift_down_indices = get_input_value_shift_down_indices(input_filename)
    input_value_shift_left_indices = get_input_value_shift_left_indices(input_filename)
    input_value_shift_right_indices = get_input_value_shift_right_indices(input_filename)
    weighted_average = get_weighted_average(weights, input_red_shift_up_indices, input_red_shift_down_indices, input_red_shift_left_indices, input_red_shift_right_indices, input_green_shift_up_indices, input_green_shift_down_indices, input_green_shift_left_indices, input_green_shift_right_indices, input_blue_shift_up_indices, input_blue_shift_down_indices, input_blue_shift_left_indices, input_blue_shift_right_indices, input_hue_shift_up_indices, input_hue_shift_down_indices, input_hue_shift_left_indices, input_hue_shift_right_indices, input_saturation_shift_up_indices, input_saturation_shift_down_indices, input_saturation_shift_left_indices, input_saturation_shift_right_indices, input_value_shift_up_indices, input_value_shift_down_indices, input_value_shift_left_indices, input_value_shift_right_indices)
    #get size of ith image
    image = cv2.imread(dataset_folder+input_filenames[i])
    height, width, channels = image.shape
    weighted_average[weighted_average>height-1] = height-1
    weighted_average[weighted_average<0] = 0
    #make black image same size as ith image
    output_image = np.zeros((height, width), np.uint8)
    #fill in black image with weighted average
    for j in range(len(weighted_average)):
        output_image[int(weighted_average[j]),j] = 255
    #save image
    cv2.imwrite('trained'+output_filenames[i], output_image)
    print(i)