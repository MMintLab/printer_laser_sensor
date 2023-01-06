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
def get_weighted_average(weights, array1, array2, array3, array4, array5, array6):
    weighted_average = (weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4 + weights[4,0]*array5 + weights[5,0]*array6)/np.sum(weights)
    return weighted_average

#concatenate all images in a filename list horizontally and if they have different heights, pad them with zeros
def concatenate_images_horizontally(filename_list, dataset_folder):
    #get first image in list
    image = cv2.imread(dataset_folder + filename_list[0])
    #get height and width of image
    height, width, channels = image.shape
    #for each image in the list, concatenate it horizontally
    for i in range(1, len(filename_list)):
        #get image
        image2 = cv2.imread(dataset_folder + filename_list[i])
        #get height and width of image
        height2, width2, channels2 = image2.shape
        #if the height of the two images are different, pad the image with zeros
        if height != height2:
            #get difference in height
            height_difference = abs(height - height2)
            #pad image with zeros
            if height > height2:
                image2 = np.pad(image2, ((0,height_difference),(0,0),(0,0)), 'constant', constant_values=(0))
            else:
                image = np.pad(image, ((0,height_difference),(0,0),(0,0)), 'constant', constant_values=(0))
        #concatenate images horizontally
        image = np.concatenate((image, image2), axis=1)
    return image

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

def get_input_red_indices(input_filenames):
    input_red_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_red_channel = get_red_channel(input_image)
        input_red_indices = np.append(input_red_indices, get_highest_intensity_pixel_indices(input_red_channel))
    return input_red_indices

def get_input_green_indices(input_filenames):
    input_green_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_green_channel = get_green_channel(input_image)
        input_green_indices = np.append(input_green_indices, get_highest_intensity_pixel_indices(input_green_channel))
    return input_green_indices

def get_input_blue_indices(input_filenames):
    input_blue_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_blue_channel = get_blue_channel(input_image)
        input_blue_indices = np.append(input_blue_indices, get_highest_intensity_pixel_indices(input_blue_channel))
    return input_blue_indices

def get_input_hue_indices(input_filenames):
    input_hue_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_hue_channel = get_hue_channel(input_image)
        input_hue_indices = np.append(input_hue_indices, get_highest_intensity_pixel_indices(input_hue_channel))
    return input_hue_indices

def get_input_saturation_indices(input_filenames):
    input_saturation_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_saturation_channel = get_saturation_channel(input_image)
        input_saturation_indices = np.append(input_saturation_indices, get_highest_intensity_pixel_indices(input_saturation_channel))
    return input_saturation_indices

def get_input_value_indices(input_filenames):
    input_value_indices = np.zeros((0,0)).astype(int)
    numfiles = len(input_filenames)
    for i in range(0, numfiles):
        input_image = cv2.imread(dataset_folder + input_filenames[i], cv2.IMREAD_COLOR)
        input_value_channel = get_value_channel(input_image)
        input_value_indices = np.append(input_value_indices, get_highest_intensity_pixel_indices(input_value_channel))
    return input_value_indices

#given increment and index, alter one element of array by increment
def alter_one_element(array, increment, index):
    array[index] = array[index] + increment
    return array

def gradient_descent(weights, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel, incr, learning_rate, output_image):
    weighted_average = get_weighted_average(weights, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)
    input_indices = get_highest_intensity_pixel_indices(weighted_average)
    output_indices = get_highest_intensity_pixel_indices(output_image)
    #find gradient of the cost for each set of indices
    
    orig_cost = compute_cost(input_indices, output_indices)
    weights1 = weights+np.array([incr,0,0,0,0,0]).reshape(6,1)
    #print(weights1)
    #input("Press Enter to continue...")
    weights2 = weights+np.array([0,incr,0,0,0,0]).reshape(6,1)
    #print(weights2)
    #input("Press Enter to continue...")
    weights3 = weights+np.array([0,0,incr,0,0,0]).reshape(6,1)
    #print(weights3)
    #input("Press Enter to continue...")
    weights4 = weights+np.array([0,0,0,incr,0,0]).reshape(6,1)
    #print(weights4)
    #input("Press Enter to continue...")
    weights5 = weights+np.array([0,0,0,0,incr,0]).reshape(6,1)
    #print(weights5)
    #input("Press Enter to continue...")
    weights6 = weights+np.array([0,0,0,0,0,incr]).reshape(6,1)
    #print(weights6)
    #input("Press Enter to continue...")
    red_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights1, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)), output_indices)
    green_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights2, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)), output_indices)
    blue_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights3, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)), output_indices)
    hue_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights4, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)), output_indices)
    saturation_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights5, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)), output_indices)
    value_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights6, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)), output_indices)
    cost_array = np.array([red_cost, green_cost, blue_cost, hue_cost, saturation_cost, value_cost])
    gradient_array = (cost_array-orig_cost)/incr
    #print(cost_array)
    #input("Press Enter to continue...")
    #print(gradient_array)
    #input("Press Enter to continue...")
    new_weights = weights - (learning_rate * incr * gradient_array.reshape(6, 1))
    #print(new_weights)
    #input("Press Enter to continue...")
    return new_weights

output_image = concatenate_images_horizontally(output_filenames, dataset_folder)
input_image = concatenate_images_horizontally(input_filenames, dataset_folder)
cv2.imshow("output", output_image)
cv2.imshow("input", input_image)
red_image = get_red_channel(input_image)
green_image = get_green_channel(input_image)
blue_image = get_blue_channel(input_image)
hue_image = get_hue_channel(input_image)
saturation_image = get_saturation_channel(input_image)
value_image = get_value_channel(input_image)

#create 6x1 numpy array of ones
weights = np.ones((6,1))
weights[1] = -1
weights[2] = -1
weights[4] = -1

#weighted average of all indices
weighted_average1 = get_weighted_average(weights, red_image, green_image, blue_image, hue_image, saturation_image, value_image)
output_indices1 = get_highest_intensity_pixel_indices(output_image)
input_indices1 = get_highest_intensity_pixel_indices(weighted_average1)
cost = compute_cost(input_indices1, output_indices1)
#print(cost)
prevcost = 2*cost

incr = 0.01

learning_rate = 0.00001

#gradient descent until cost changes by less than 1 percent
while abs(cost-prevcost) > 0.0000000001*prevcost:
    print(cost)
    prevcost = cost
    weights = gradient_descent(weights, red_image, green_image, blue_image, hue_image, saturation_image, value_image, incr, learning_rate, output_image)
    weighted_average1 = get_weighted_average(weights, red_image, green_image, blue_image, hue_image, saturation_image, value_image)
    output_indices1 = get_highest_intensity_pixel_indices(output_image)
    input_indices1 = get_highest_intensity_pixel_indices(weighted_average1)
    cost = compute_cost(input_indices1, output_indices1)

print(weights.reshape(6,1))

#test on test images
for i in range(len(input_filenames)):
    #make a list containing only the ith element of input_filenames
    input_filename = [input_filenames[i]]
    red_channel = get_red_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    green_channel = get_green_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    blue_channel = get_blue_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    hue_channel = get_hue_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    saturation_channel = get_saturation_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    value_channel = get_value_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    weighted_average = get_weighted_average(weights, red_channel, green_channel, blue_channel, hue_channel, saturation_channel, value_channel)
    #print(weighted_average.shape)
    #print(np.max(weighted_average))
    #print(np.min(weighted_average))
    indices = get_highest_intensity_pixel_indices(weighted_average)
    #print(indices)
    #get size of ith image
    image = cv2.imread(dataset_folder+input_filenames[i])
    height, width, channels = image.shape
    #make black image same size as ith image
    output_image1 = np.zeros((height, width,channels), np.uint8)
    #fill in black image with weighted average
    for j in range(indices.shape[1]):
        output_image1[int(indices[0,j]), j,:] = 255
    #save image
    cv2.imwrite('trained'+output_filenames[i], output_image1)
    #print(i)