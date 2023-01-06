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

#alter grayscale image from 0-255 to 0-1
def alter_grayscale_image(imagegray):
    imagegray = imagegray/255
    return imagegray

#append two images vertically
def append_images_vertically(image1, image2):
    image = np.concatenate((image1, image2), axis=0)
    return image

#get weighted average of 24 arrays
def get_weighted_average(weights, array1, array2, array3, array4, array5, array6, array7, array8, array9):
    weighted_average = (weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4 + weights[4,0]*array5 + weights[5,0]*array6 + weights[6,0]*array7 + weights[7,0]*array8 + weights[8,0]*array9)/np.sum(weights)
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

#binary cross entropy loss function
def binary_cross_entropy_loss_function(array2, array1):
    #limit array2 values to be between 0 and 1
    array2 = np.clip(array2, 0.0001, 0.9999)
    #print(np.max(array1))
    #print(np.max(array2))
    #print(np.min(array1))
    #print(np.min(array2))
    #print number of unique values in array1
    #print(np.unique(array1))
    #print number of unique values in array2
    #print(np.unique(array2))
    loss = -np.sum(array1*np.log(array2) + (1-array1)*np.log(1-array2))
    return loss

#compute cost given two arrays
def compute_cost(array_in, array_out):
    #difference between array_in adjacent elements
    array_in_diff = np.diff(array_in)
    #difference between array_out adjacent elements
    array_out_diff = np.diff(array_out)
    cost = 1*np.sum(np.square(array_in - array_out)) + 1*np.sum(np.square(array_in_diff - array_out_diff)) + 0*np.sum(np.square(array_in_diff))
    return cost

def get_highest_intensity_pixel_indices(image):
    highest_intensity_pixel_indices = np.zeros((1,image.shape[1]))
    for i in range(image.shape[1]):
        highest_intensity_pixel_indices[0,i] = np.sum((image[:,i]*np.linspace(start=0, stop=image.shape[0]-1, num=image.shape[0], axis=0)))/np.sum(image[:,i])
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

def gradient_descent(weights, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel, incr, learning_rate, output_image):
    weighted_average = get_weighted_average(weights, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)
    #find gradient of the cost for each set of indices
    
    orig_cost = binary_cross_entropy_loss_function(alter_grayscale_image(weighted_average), alter_grayscale_image(output_image))
    #weights1 = weights+np.array([incr,0,0,0,0,0,0,0,0]).reshape(9,1)
    weights2 = weights+np.array([0,incr,0,0,0,0,0,0,0]).reshape(9,1)
    weights3 = weights+np.array([0,0,incr,0,0,0,0,0,0]).reshape(9,1)
    weights4 = weights+np.array([0,0,0,incr,0,0,0,0,0]).reshape(9,1)
    weights5 = weights+np.array([0,0,0,0,incr,0,0,0,0]).reshape(9,1)
    weights6 = weights+np.array([0,0,0,0,0,incr,0,0,0]).reshape(9,1)
    weights7 = weights+np.array([0,0,0,0,0,0,incr,0,0]).reshape(9,1)
    weights8 = weights+np.array([0,0,0,0,0,0,0,incr,0]).reshape(9,1)
    weights9 = weights+np.array([0,0,0,0,0,0,0,0,incr]).reshape(9,1)

    #red_cost = compute_cost(get_highest_intensity_pixel_indices(get_weighted_average(weights1, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), output_indices)
    green_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights2, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    blue_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights3, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    left_red_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights4, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    left_green_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights5, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    left_blue_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights6, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    right_red_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights7, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    right_green_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights8, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))
    right_blue_cost = binary_cross_entropy_loss_function(alter_grayscale_image(get_weighted_average(weights9, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)), alter_grayscale_image(output_image))

    cost_array = np.array([orig_cost, green_cost, blue_cost, left_red_cost, left_green_cost, left_blue_cost, right_red_cost, right_green_cost, right_blue_cost])
    gradient_array = (cost_array-orig_cost)/incr
    #print(cost_array)
    #input("Press Enter to continue...")
    #print(gradient_array)
    #input("Press Enter to continue...")
    new_weights = weights - (learning_rate * incr * gradient_array.reshape(9, 1))
    #print(new_weights)
    #input("Press Enter to continue...")
    return new_weights

output_image0 = concatenate_images_horizontally(output_filenames, dataset_folder)
#make output image grayscale
output_image = cv2.cvtColor(output_image0, cv2.COLOR_BGR2GRAY)
input_image = concatenate_images_horizontally(input_filenames, dataset_folder)
cv2.imshow("output", output_image)
cv2.imshow("input", input_image)
red_image = get_red_channel(input_image)
green_image = get_green_channel(input_image)
blue_image = get_blue_channel(input_image)
left_red_image = shift_image_left(red_image)
left_green_image = shift_image_left(green_image)
left_blue_image = shift_image_left(blue_image)
right_red_image = shift_image_right(red_image)
right_green_image = shift_image_right(green_image)
right_blue_image = shift_image_right(blue_image)

#create 6x1 numpy array of ones
weights = np.ones((9,1))
weights[0] = 1
weights[1] = 0.159
weights[2] = -0.447

weights[3] = 1.060
weights[4] = -0.303
weights[5] = -0.490

weights[6] = 0.79
weights[7] = -0.3
weights[8] = 0.3

#make random weights
#weights = np.random.rand(9,1)

#weighted average of all indices
weighted_average1 = get_weighted_average(weights, red_image, green_image, blue_image, left_red_image, left_green_image, left_blue_image, right_red_image, right_green_image, right_blue_image)
output_indices1 = get_highest_intensity_pixel_indices(output_image)
input_indices1 = get_highest_intensity_pixel_indices(weighted_average1)
cost = binary_cross_entropy_loss_function(alter_grayscale_image(weighted_average1), alter_grayscale_image(output_image))
#print(cost)
prevcost = 2*cost

incr = 0.01

learning_rate = 0.001

#gradient descent until cost changes by less than 1 percent
try:
    while abs(cost-prevcost) > 0.0001*prevcost:
        #print cost and weights
        print(cost)
        print(weights.reshape(1,9))
        prevcost = cost
        weights = gradient_descent(weights, red_image, green_image, blue_image, left_red_image, left_green_image, left_blue_image, right_red_image, right_green_image, right_blue_image, incr, learning_rate, output_image)
        weighted_average1 = get_weighted_average(weights, red_image, green_image, blue_image, left_red_image, left_green_image, left_blue_image, right_red_image, right_green_image, right_blue_image)
        output_indices1 = get_highest_intensity_pixel_indices(output_image)
        input_indices1 = get_highest_intensity_pixel_indices(weighted_average1)
        cost = binary_cross_entropy_loss_function(alter_grayscale_image(weighted_average1),alter_grayscale_image(output_image))
except KeyboardInterrupt:
    print("Interrupted")
    pass

print(weights.reshape(9,1))

#test on test images
for i in range(len(input_filenames)):
    #make a list containing only the ith element of input_filenames
    input_filename = [input_filenames[i]]
    red_channel = get_red_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    green_channel = get_green_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    blue_channel = get_blue_channel(concatenate_images_horizontally(input_filename, dataset_folder))
    left_red_channel = shift_image_left(red_channel)
    left_green_channel = shift_image_left(green_channel)
    left_blue_channel = shift_image_left(blue_channel)
    right_red_channel = shift_image_right(red_channel)
    right_green_channel = shift_image_right(green_channel)
    right_blue_channel = shift_image_right(blue_channel)

    weighted_average = get_weighted_average(weights, red_channel, green_channel, blue_channel, left_red_channel, left_green_channel, left_blue_channel, right_red_channel, right_green_channel, right_blue_channel)
    #stack image on top of itself depth
    indices = get_highest_intensity_pixel_indices(weighted_average)
    #limit indices to zero
    indices[indices < 0] = 0
    #limit indices to height of weighted_average
    indices[indices > weighted_average.shape[0]-1] = weighted_average.shape[0]-1
    #get size of ith image
    image = cv2.imread(dataset_folder+input_filenames[i])
    height, width, channels = image.shape
    #make black image same size as ith image
    output_image1 = np.zeros((height, width,channels), np.uint8)
    #fill in black image with weighted average
    weighted_average = np.dstack((weighted_average, weighted_average, weighted_average))
    for j in range(indices.shape[1]):
        output_image1[int(indices[0,j]), j,:] = 255
    #save image
    cv2.imwrite('trained'+output_filenames[i], append_images_vertically(append_images_vertically(image, weighted_average), output_image1))
    #print(i)