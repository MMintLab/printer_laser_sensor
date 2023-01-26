from time import sleep
import time
from PIL import Image
import PIL.ImageOps as imops
import io
import cv2 as cv
import numpy as np
import sys
import os

test_number = 12#int(sys.argv[1])

data_folder = 'data/Jan11_2023'

#get green channel of image
def get_green_channel(image):
    #convert image to numpy array
    image_array = np.array(image)
    #get green channel
    green_channel = image_array[:,:,1]
    return green_channel

def get_red_channel(image):
    #convert image to numpy array
    image_array = np.array(image)
    #get red channel
    red_channel = image_array[:,:,0]
    return red_channel

def get_blue_channel(image):
    #convert image to numpy array
    image_array = np.array(image)
    #get blue channel
    blue_channel = image_array[:,:,2]
    return blue_channel

#get list of all files in data_foler beginning with test_[test_number] and ending with .jpg
def get_filename_list(test_number,data_folder):
    image_list = []
    #print('test%02d' % (test_number))
    for filename in os.listdir(data_folder):
        if filename.startswith(str('test%02d' % (test_number))) and filename.endswith('.jpg'):
            image_list.append(filename)

    #sort list of images
    image_list.sort()
    return image_list
    

image_list = get_filename_list(test_number,data_folder)
print(image_list)

#open images in image_list and show green channel
for i in range(len(image_list)):
    filename = image_list[i]
    image = Image.open(data_folder+'/'+filename)
    cv.imshow('green channel',get_green_channel(image))
    cv.imshow('red channel',get_red_channel(image))
    cv.imshow('blue channel',get_blue_channel(image))
    cv.waitKey(0)

cv.destroyAllWindows()
