import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

dataset_folder = 'nozzleseg_dataset_v2/'

#get sorted list of filenames beginning with 'input'
input_filenames = sorted([img for img in os.listdir(dataset_folder) if img.startswith('input')])
#get sorted list of filenames beginning with 'output'
output_filenames = sorted([img for img in os.listdir(dataset_folder) if img.startswith('output')])

#eliminate all elements of input_filenames except for the first five
#input_filenames = input_filenames[0:9]
#eliminate all elements of output_filenames except for the first five
#output_filenames = output_filenames[0:9]

def get_factors(number):
    factors = []
    for i in range(1,number+1):
        if number%i==0:
            factors.append(i)
    return factors

def get_closest_factors(factors):
    #get the two factors closest to each other
    closest_factors = []
    closest_factors.append(factors[int(len(factors)/2)])
    closest_factors.append(factors[int(len(factors)/2)-1])
    return closest_factors

#load all images and concatenate into the squarest grid possible given the number of images
def load_images(input_filenames, output_filenames, dataset_folder):
    #get list of factors of the number of images
    factors = get_factors(len(input_filenames))
    #get the two factors closest to each other
    factors = get_closest_factors(factors)
    #if the number of images is a perfect square, use the square root as the number of rows and columns
    if np.sqrt(len(input_filenames))%1==0:
        factors[0] = int(np.sqrt(len(input_filenames)))
        factors[1] = int(np.sqrt(len(input_filenames)))
    #get the number of rows and columns
    rows = factors[0]
    cols = factors[1]
    #get size of each image
    image = cv2.imread(dataset_folder+input_filenames[0])
    image_size = image.shape
    #initialize input and output arrays
    input_array = np.zeros((image_size[0]*rows, image_size[1]*cols, image_size[2]))
    output_array = np.zeros((image_size[0]*rows, image_size[1]*cols, image_size[2]))
    #load images into input and output arrays
    for i in range(len(input_filenames)):
        input_image = cv2.imread(dataset_folder+input_filenames[i])
        output_image = cv2.imread(dataset_folder+output_filenames[i])
        #get row and column of image
        row = int(i/cols)
        col = i%cols
        #add image to input and output arrays
        #print(input_array.shape)
        input_array[row*image_size[0]:(row+1)*image_size[0], col*image_size[1]:(col+1)*image_size[1], :] = input_image
        output_array[row*image_size[0]:(row+1)*image_size[0], col*image_size[1]:(col+1)*image_size[1], :] = output_image

    input_array = input_array.astype(np.uint8)
    output_array = output_array.astype(np.uint8)
    return input_array, output_array

def model_output(input_image, weights, threshold):
    red_channel = get_red_channel(input_image)
    green_channel = get_green_channel(input_image)
    blue_channel = get_blue_channel(input_image)
    sat_channel = get_sat_channel(input_image)
    value_channel = get_value_channel(input_image)
    #make all images go from 0 to 1
    red_channel = red_channel/255
    green_channel = green_channel/255
    blue_channel = blue_channel/255
    sat_channel = sat_channel/255
    value_channel = value_channel/255
    weighted_average = get_weighted_average(weights, red_channel, green_channel, blue_channel, sat_channel, value_channel)
    #output image is 255 if weighted average is greater than threshold, 0 otherwise
    output_image = weighted_average
    return output_image

#get weighted average of 24 arrays
def get_weighted_average(weights, array1, array2, array3, array4, array5):
    weighted_average = weights[3,0]*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3)/(np.sum(weights))+weights[4,0]*array4*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3)/(np.sum(weights))
    return weighted_average

#compute cost given two arrays
def compute_cost(array1, array2):
    array1 = np.clip(array1, 0.0001, 0.9999)
    cost = -np.sum(array2*np.log(array1) + (1-array2)*np.log(1-array1))
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

def get_sat_channel(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_channel = hsv_image[:,:,1]
    return sat_channel

def get_value_channel(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:,:,2]
    return value_channel

def gradient_descent(input_image, output_image, weights, threshold, incr, learning_rate):
    output_test = model_output(input_image, weights, threshold)
    #find gradient of the cost for each set of indices
    orig_cost = compute_cost(output_test, output_image)
    weights1 = weights+np.array([incr,0,0,0,0]).reshape(5,1)
    weights2 = weights+np.array([0,incr,0,0,0]).reshape(5,1)
    weights3 = weights+np.array([0,0,incr,0,0]).reshape(5,1)
    weights4 = weights+np.array([0,0,0,incr,0]).reshape(5,1)
    weights5 = weights+np.array([0,0,0,0,incr]).reshape(5,1)
    cost1 = compute_cost(model_output(input_image, weights1, threshold), output_image)
    cost2 = compute_cost(model_output(input_image, weights2, threshold), output_image)
    cost3 = compute_cost(model_output(input_image, weights3, threshold), output_image)
    cost4 = compute_cost(model_output(input_image, weights4, threshold), output_image)
    cost5 = compute_cost(model_output(input_image, weights5, threshold), output_image)
    cost_array = np.array([cost1, cost2, cost3, cost4, cost5])
    gradient_array = (cost_array-orig_cost)/incr

    #update weights and threshold
    new_weights = weights - learning_rate*gradient_array.reshape(5,1)
    #print(new_weights)
    #input("Press Enter to continue...")
    return new_weights

#create 2x1 random numpy array
weights = np.random.rand(5,1)

#[[-0.36273123]
# [ 1.26436845]
# [-0.93253676]
# [ 0.27426181]
# [-0.2161281 ]]

#initialize random threshold between 0 and 255
threshold = np.random.randint(0,256)

input_image, output_image = load_images(input_filenames=input_filenames, output_filenames=output_filenames, dataset_folder=dataset_folder)

#convert output image to grayscale
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
#make output image go from 0 to 1
output_image = output_image/255

#weighted average of all indices
output_test = model_output(input_image, weights, threshold)

cost = compute_cost(output_test, output_image)
#print(cost)
prevcost = 2*cost

incr = 0.0001

learning_rate = 0.0000003

#gradient descent until cost changes by less than 1 percent
while abs(cost-prevcost) > 0.0001*prevcost:
    print(cost)
    print(weights.reshape(5,1))
    prevcost = cost
    weights = gradient_descent(input_image, output_image, weights, threshold, incr, learning_rate)
    output_test = model_output(input_image, weights, threshold)
    #print(np.max(output_test))
    #print(np.min(output_test))
    #show output image
    cv2.imshow('output', output_test)
    cv2.waitKey(1)
    cost = compute_cost(output_test, output_image)

print(cost)
print(weights.reshape(5,1))
#test on test images
for i in range(len(input_filenames)):
    #make a list containing only the ith element of input_filenames
    image = cv2.imread(dataset_folder+input_filenames[i])
    #make image go from 0 to 1
    height, width, channels = image.shape
    output_test = model_output(image, weights, threshold)
    #show output image
    cv2.imshow('output', output_test)
    cv2.waitKey(0)
    #clip output image to be between 0 and 1
    output_test = np.clip(output_test, 0, 1)
    #make output image go from 0 to 255
    output_test = (output_test*255).astype(np.uint8)
    cv2.imwrite('trained'+output_filenames[i], output_test)
    #print(i)