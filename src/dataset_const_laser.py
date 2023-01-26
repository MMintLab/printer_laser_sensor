import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

data_folder = 'data/Jan11_2023/'

#create global variables mouseX and mouseY as empty lists
global mouseX
mouseX = []
global mouseY
mouseY = []

def rotate_image_180(image):
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    return rotated_image

def crop_image_horizontally(image, percent):
    height, width, channels = image.shape
    image = image[0:height, int(width/2 - width*percent/2):int(width/2 + width*percent/2)]
    return image

def make_image_bigger(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

#first argument from terminal is the number of images to select
num_images = int(sys.argv[1])

def focus_on_lower_part_percentage(image, percentage):
    height, width, channels = image.shape
    image = image[height-int(height*percentage):height, 0:width]
    return image

#given list of four mouse click coordinates, mouseX and mouseY, and an image, output an image with a quadrilateral filled in white
def fill_quadrilateral(image, mouseX1, mouseY1):
    #create a black image
    black_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    #draw a white quadrilateral on the black image
    cv2.fillPoly(black_image, np.array([[[mouseX1[0], mouseY1[0]], [mouseX1[1], mouseY1[1]], [mouseX1[2], mouseY1[2]], [mouseX1[3], mouseY1[3]]]], dtype=np.int32), (255, 255, 255))
    return black_image

#focus on upper part of image based on given percentage
def focus_on_upper_part_percentage(image, percentage):
    height, width, channels = image.shape
    image = image[0:int(height*percentage), 0:width]
    return image

#select a random set of images in a data folder
def select_random_images(data_folder, num_images):
    images = [img for img in os.listdir(data_folder) if img.endswith('.jpg')]
    #select random images from the set
    random_images = np.random.choice(images, num_images, replace=False)
    return random_images

#return left mouse click position over image
def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #append x to the list mouseX
        mouseX.append(x)
        #append y to the list mouseY
        mouseY.append(y)
        #print mouseX and mouseY
        print(mouseX, mouseY)

#record mouse click position as white pixels in black image
def record_mouse_click(data_folder, filename_list):
    i = 0
    for filename in filename_list:
        image = cv2.imread(data_folder + filename)
        #save image
        cv2.imwrite('input_' + filename, image)
        i = i + 1

filename_list = select_random_images(data_folder, num_images)
print(filename_list)
record_mouse_click(data_folder, filename_list)