import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

data_folder = 'data/Jan11_2023'

test_number = int(sys.argv[1])

#get sorted list of all files in data_foler beginning with test_[test_number] and ending with .jpg
def get_filename_list(test_number,data_folder):
    image_list = []
    #print('test%02d' % (test_number))
    for filename in os.listdir(data_folder):
        if filename.startswith(str('test%02d' % (test_number))) and filename.endswith('.jpg'):
            image_list.append(filename)
    
    #sort list of images
    image_list.sort()
    return image_list

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

def model_output(input_image, weights):
    red_channel = get_red_channel(input_image)
    green_channel = get_green_channel(input_image)
    blue_channel = get_blue_channel(input_image)
    sat_channel = get_sat_channel(input_image)
    value_channel = get_value_channel(input_image)
    lum_channel = get_lum_channel(input_image)
    #make all images go from 0 to 1
    red_channel = red_channel/255
    green_channel = green_channel/255
    blue_channel = blue_channel/255
    sat_channel = sat_channel/255
    value_channel = value_channel/255
    lum_channel = lum_channel/255
    weighted_average = get_weighted_average(weights, red_channel, green_channel, blue_channel, sat_channel, value_channel, lum_channel)
    #output image is 255 if weighted average is greater than threshold, 0 otherwise
    output_image = weighted_average
    return output_image

#get weighted average of 24 arrays
#def get_weighted_average(weights, array1, array2, array3, array4, array5, array6):
    #weighted_average = weights[3,0]*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3)/(np.sum(weights))+weights[4,0]*array4*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3)/(np.sum(weights))+weights[5,0]*array5*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3)/(np.sum(weights))
    #weighted_average = weights[4,0]*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4)/(np.sum(weights))+weights[5,0]0*array5*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4)/(np.sum(weights))+weights[6,0]*array6*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4)/(np.sum(weights))
#    weighted_average = array6*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4 + weights[4,0]*array5 + weights[5,0]*array4*array5 + weights[6,0]*array4*array6)
    #weighted_average = (1+weights[5,0]*array6)*(weights[0,0]*array1 + weights[1,0]*array2 + weights[2,0]*array3 + weights[3,0]*array4 + weights[4,0]*array5)/(np.sum(weights))
#    return weighted_average

def get_weighted_average(dumb,R,G,B,S,V,L):
    weights = np.array([[2.77323259],[-4.28669185],[2.87523104],[-0.40948264],[-2.14548059],[0.69278767],[1.30010453]])
    output = L*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B + weights[3,0]*S + weights[4,0]*V + weights[5,0]*S*V + weights[6,0]*S*L)
    #clip output to be between 0 and 1
    #np.clip(output, 0, 1)
    return output

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
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    sat_channel = hsv_image[:,:,2]
    return sat_channel

def get_lum_channel(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lum_channel = hsv_image[:,:,1]
    return lum_channel

def get_value_channel(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:,:,2]
    return value_channel

#create 2x1 random numpy array
weights = np.random.rand(7,1)

weights[0,0] = 2.77323259
weights[1,0] = -4.28669185
weights[2,0] = 2.87523104
weights[3,0] = -0.40948264
weights[4,0] = -2.14548059
weights[5,0] = 0.69278767
weights[6,0] = 1.30010453

filename_list = get_filename_list(test_number,data_folder)

def save_video_mp4(image_folder, testnumber):
    video_name = 'test%02d_laserseg.mp4' % (testnumber)
    images = [img for img in os.listdir(image_folder) if img.startswith(str('test%02d' % (testnumber))) and img.endswith(".jpg")]
    print(images)
    #sort images by number i.e. test01camera1image00001.jpeg, test01camera1image00002.jpeg, etc.
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width*2,height))

    for image in images:
        input_image = cv2.imread(os.path.join(image_folder, image))
        output_test = model_output(input_image, weights)
        output_test = np.clip(output_test, 0, 1)
        output_test = (output_test*255).astype(np.uint8)
        output_test = cv2.cvtColor(output_test, cv2.COLOR_GRAY2BGR)
        #concatenate input and output images
        image_concat = np.concatenate((input_image, output_test), axis=1)
        video.write(image_concat)
    video.release()

save_video_mp4(data_folder, test_number)