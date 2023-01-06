import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

data_folder = '/Users/william/Documents/MMINT_Research/printer_laser_sensor/data/Dec15_2022/'

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
    global mouseX
    global mouseY
    scale = 20
    i = 0
    for filename in filename_list:
        #check if filename includes 'camera1' or 'camera0'
        if 'camera0' in filename:
            image = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(rotate_image_180(cv2.imread(data_folder + filename)),0.42), 0.39),0.2)
        elif 'camera1' in filename:
            image = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(rotate_image_180(cv2.imread(data_folder + filename)),0.38), 0.4),0.2)
        imageBIG = make_image_bigger(image, scale*100)
        cv2.namedWindow('imageCropCheck')
        cv2.imshow('imageCropCheck', imageBIG)
        cv2.waitKey(10)
        wrong_crop = input("Is the image cropped correctly? (y/n): ")
        if wrong_crop == 'n':
            if 'camera1' in filename:
                image = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(rotate_image_180(cv2.imread(data_folder + filename)),0.42), 0.39),0.2)
            elif 'camera0' in filename:
                image = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(rotate_image_180(cv2.imread(data_folder + filename)),0.38), 0.4),0.2)
        imageBIG = make_image_bigger(image, scale*100)
        cv2.namedWindow('laserCheck')
        cv2.imshow('laserCheck', imageBIG)
        cv2.waitKey(10)
        wrong_laser = input("Is the laser visible? (y/n): ")
        if wrong_laser == 'n':
            print('disregarding image')
            cv2.destroyAllWindows()
            continue
        cv2.destroyAllWindows()
        imageBIG = make_image_bigger(image, scale*100)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', get_mouse_click)
        cv2.imshow('image', imageBIG)
        cv2.waitKey(0)
        #print mouse click position
        mouseX = [int(x/scale) for x in mouseX]
        mouseY = [int(y/scale) for y in mouseY]
        print(mouseX, mouseY)
        cv2.namedWindow('selected')
        image_selected = image[:,:,0]*0
        image_selected[mouseY, mouseX] = 255
        cv2.imshow('selected', make_image_bigger(image_selected, scale*100))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #reset mouseX and mouseY
        mouseX = []
        mouseY = []
        #save image
        cv2.imwrite('input_' + filename, image)
        cv2.imwrite('output_' + filename, image_selected)
        i = i + 1

filename_list = select_random_images(data_folder, num_images)
print(filename_list)
record_mouse_click(data_folder, filename_list)