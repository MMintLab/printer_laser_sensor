import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

data_folder = '/Users/william/Documents/MMINT_Research/printer_laser_sensor/data/Dec15_2022/'

testnumber = 3

frontcameranumber = 1
rearcameranumber = 0

#give indices of highest intensity pixels as integers in each column of an image
def get_highest_intensity_pixel_indices(image):
    highest_intensity_pixel_indices = np.zeros((image.shape[1], 2)).astype(int)
    for i in range(image.shape[1]):
        highest_intensity_pixel_indices[i,0] = np.argmax(image[:,i])
        highest_intensity_pixel_indices[i,1] = i
    return highest_intensity_pixel_indices

#scale image pixel values to 0-255
def image_255(image):
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)
    return image

#make image bigger
def make_image_bigger(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

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
    saturation_channel = image[:,:,1]
    return saturation_channel

def get_value_channel(image):
    value_channel = image[:,:,2]
    return value_channel

def get_hue_channel(image):
    hue_channel = image[:,:,0]
    return hue_channel

def rotate_image_180(image):
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    return rotated_image

def get_num_frames(folder,testnumber,cameranumber):
    num_frames = 0
    for file in os.listdir(folder):
        if file.startswith(str('test%02dcamera%dimage' % (testnumber, cameranumber))):
            num_frames += 1
    return num_frames

def convert_image_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

#get ordered list of filenames based on testnumber and cameranumber
def get_ordered_filenames(image_folder,testnumber,cameranumber):
    images = [img for img in os.listdir(image_folder) if img.startswith(str('test%02dcamera%dimage' % (testnumber, cameranumber)))]
    #sort images by number i.e. test01camera1image00001.jpeg, test01camera1image00002.jpeg, etc.
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return images

front_num_frames = get_num_frames(data_folder,testnumber,frontcameranumber)

rear_num_frames = get_num_frames(data_folder,testnumber,rearcameranumber)

#focus on lower part of image based on given percentage
def focus_on_lower_part_percentage(image, percentage):
    height, width, channels = image.shape
    image = image[height-int(height*percentage):height, 0:width]
    return image

#focus on upper part of image based on given percentage
def focus_on_upper_part_percentage(image, percentage):
    height, width, channels = image.shape
    image = image[0:int(height*percentage), 0:width]
    return image

#create an array of white pixel coordinates for given binary mask
def get_white_pixel_coordinates(mask):
    white_pixel_coordinates = np.argwhere(mask)
    return white_pixel_coordinates

#focus on lower part of image based on given pixel count
def focus_on_lower_part(image, pixel_count):
    height, width, channels = image.shape
    image = image[height-pixel_count:height, 0:width]
    return image

#edge detection
def edge_detection(image):
    edges = cv2.Canny(image,100,200)
    return edges

#crop image horizontally around center by percent
def crop_image_horizontally(image, percent):
    height, width, channels = image.shape
    image = image[0:height, int(width/2 - width*percent/2):int(width/2 + width*percent/2)]
    return image

#make image grayscale
def make_image_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

#organize datapoints in numpy array based on horizontal coordinate and find the average vertical coordinate for each horizontal coordinate
def organize_datapoints_by_horizontal_coordinate(datapoints):
    horizontal_coordinates = np.unique(datapoints[:,1])
    organized_datapoints = np.zeros((len(horizontal_coordinates), 2))
    for i in range(len(horizontal_coordinates)):
        organized_datapoints[i,1] = horizontal_coordinates[i]
        organized_datapoints[i,0] = np.mean(datapoints[datapoints[:,1] == horizontal_coordinates[i],0])
    return organized_datapoints

#integrate discrete datapoints in numpy array
def integrate_datapoints(datapoints):
    integrated_datapoints = np.zeros((len(datapoints), 2))
    integrated_datapoints[0,0] = datapoints[0,0]
    integrated_datapoints[0,1] = datapoints[0,1]
    for i in range(1,len(datapoints)):
        integrated_datapoints[i,0] = integrated_datapoints[i-1,0] + datapoints[i,0]
        integrated_datapoints[i,1] = datapoints[i,1]
    return integrated_datapoints

#sobel edge detection
def sobel_edge_detection(image):
    edgesX = cv2.Sobel(make_image_grayscale(image),cv2.CV_64F,1,0,ksize=5)
    edgesY = cv2.Sobel(make_image_grayscale(image),cv2.CV_64F,0,1,ksize=5)
    intensity = np.sqrt(np.square(edgesX) + np.square(edgesY))
    angle = np.arctan2(edgesY, edgesX)
    angle[angle<0] = angle[angle<0] + 3.141592653
    edges = np.zeros_like(angle)
    edges[intensity > 0.5] = 1
    edges_angle = edges*(-0.2+1.4*angle/(3.141592653))
    return edges_angle

#extract a red line from an image
def extract_red_line(image):
    hsv_image = convert_image_to_hsv(image)
    #define the range of red color in HSV
    lower_red = np.array([160, 80, 160])
    upper_red = np.array([200, 100, 255])
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    return res

def invert_image(image):
    inverted_image = 255 - image
    return inverted_image

#remove small connected areas from binary mask
def remove_small_connected_areas(mask):
    #remove small connected areas
    kernel = np.ones((10,10),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

#smooth image
def smooth_image(image):
    kernel = np.ones((10,10),np.float32)/25
    smoothed_image = cv2.filter2D(image,-1,kernel)
    return smoothed_image

front_filename_list = get_ordered_filenames(data_folder,testnumber,frontcameranumber)
rear_filename_list = get_ordered_filenames(data_folder,testnumber,rearcameranumber)

front_area = np.zeros((front_num_frames, 1))
for i in range(front_num_frames):
    front_frame_path = data_folder + front_filename_list[i]
    front_frame = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(rotate_image_180(cv2.imread(front_frame_path)),0.38), 0.4),0.2)
    hsv_front_frame = convert_image_to_hsv(front_frame)
    red_front_frame = get_red_channnel(front_frame)
    green_front_frame = get_green_channel(front_frame)
    blue_front_frame = get_blue_channel(front_frame)
    saturation_front_frame = get_saturation_channel(hsv_front_frame)
    value_front_frame = get_value_channel(hsv_front_frame)
    hue_front_frame = get_hue_channel(hsv_front_frame)
    redline_front_frame = extract_red_line(front_frame)
    cleaned_front_frame = smooth_image(remove_small_connected_areas(redline_front_frame))
    edge_detected_front_frame = edge_detection(cleaned_front_frame)
    #coordinates = get_white_pixel_coordinates(edge_detected_front_frame)
    #organized_coordinates = organize_datapoints_by_horizontal_coordinate(coordinates)
    #plt.figure(1)
    #plt.scatter(coordinates[:,1], coordinates[:,0], s=1, c='r', marker="s", label='redline')
    #plt.plot(organized_coordinates[:,1], organized_coordinates[:,0], c='m', label='redline')
    #organized_coordinates[:,0] = organized_coordinates[:,0] - organized_coordinates[0,0]
    #integrated_coordinates = integrate_datapoints(organized_coordinates)
    #plt.plot(integrated_coordinates[:,1], integrated_coordinates[:,0], c='y', label='redline')
    #plt.show(block=False)
    #input('Press Enter to continue...')
    inverted_blue_front_frame = invert_image(blue_front_frame)
    inverted_green_front_frame = invert_image(green_front_frame)
    inverted_sat_front_frame = invert_image(saturation_front_frame)
    line_red_front_frame_indices = get_highest_intensity_pixel_indices(red_front_frame)
    line_blue_front_frame_indices = get_highest_intensity_pixel_indices(inverted_blue_front_frame)
    line_green_front_frame_indices = get_highest_intensity_pixel_indices(inverted_green_front_frame)
    #print(line_red_front_frame_indices)
    line_value_front_frame_indices = get_highest_intensity_pixel_indices(value_front_frame)
    line_inverted_sat_front_frame_indices = get_highest_intensity_pixel_indices(inverted_sat_front_frame)
    line_hue_front_frame_indices = get_highest_intensity_pixel_indices(hue_front_frame)
    average_front_frame_indices = np.zeros((line_red_front_frame_indices.shape[0], 2))
    average_front_frame_indices[:,0] = (line_red_front_frame_indices[:,0] + line_blue_front_frame_indices[:,0] + line_green_front_frame_indices[:,0] + line_value_front_frame_indices[:,0] + line_inverted_sat_front_frame_indices[:,0] + line_hue_front_frame_indices[:,0])/6
    average_front_frame_indices[:,1] = (line_red_front_frame_indices[:,1])
    plt.figure(1)
    #set axis limits to be the same as the size of the image
    plt.plot(average_front_frame_indices[:,1], average_front_frame_indices[:,0], c='c')
    plt.xlim(0, front_frame.shape[1])
    plt.ylim(0, front_frame.shape[0])
    plt.show(block=False)
    line_red_front_frame = front_frame+0
    line_value_front_frame = line_red_front_frame+0
    line_inverted_sat_front_frame = line_value_front_frame+0
    line_hue_front_frame = line_inverted_sat_front_frame+0
    line_blue_front_frame = line_hue_front_frame+0
    line_green_front_frame = line_blue_front_frame+0
    line_red_front_frame[line_red_front_frame_indices[:,0], line_red_front_frame_indices[:,1]] = [0,255,255]
    line_value_front_frame[line_value_front_frame_indices[:,0], line_value_front_frame_indices[:,1]] = [0,255,255]
    line_inverted_sat_front_frame[line_inverted_sat_front_frame_indices[:,0], line_inverted_sat_front_frame_indices[:,1]] = [0,255,255]
    line_hue_front_frame[line_hue_front_frame_indices[:,0], line_hue_front_frame_indices[:,1]] = [0,255,255]
    line_blue_front_frame[line_blue_front_frame_indices[:,0], line_blue_front_frame_indices[:,1]] = [0,255,255]
    line_green_front_frame[line_green_front_frame_indices[:,0], line_green_front_frame_indices[:,1]] = [0,255,255]
    cv2.imshow('front_red', make_image_bigger(line_red_front_frame,1000))
    cv2.imshow('front_hue', make_image_bigger(line_hue_front_frame,1000))
    cv2.imshow('front_value', make_image_bigger(line_value_front_frame,1000))
    cv2.imshow('front_sat', make_image_bigger(line_inverted_sat_front_frame,1000))
    cv2.imshow('front_blue', make_image_bigger(line_blue_front_frame,1000))
    cv2.imshow('front_green', make_image_bigger(line_green_front_frame,1000))
    cv2.waitKey(1)

    #print(i)
    #front_area[i] = integrated_coordinates[-1,0]
    #cv2.waitKey(200)
    #time.sleep(0.)
    input('Press Enter to continue...')
    plt.close()

#iterate through the rear frames only and show each of them with a pause in between
rear_area = np.zeros((rear_num_frames, 1))
for i in range(rear_num_frames):
    rear_frame_path = data_folder + rear_filename_list[i]
    rear_frame = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(rotate_image_180(cv2.imread(rear_frame_path)),0.42), 0.39),0.2)
    hsv_rear_frame = convert_image_to_hsv(rear_frame)
    red_rear_frame = get_red_channnel(rear_frame)
    blue_rear_frame = get_blue_channel(rear_frame)
    green_rear_frame = get_green_channel(rear_frame)
    saturation_rear_frame = get_saturation_channel(hsv_rear_frame)
    value_rear_frame = get_value_channel(hsv_rear_frame)
    hue_rear_frame = get_hue_channel(hsv_rear_frame)
    redline_rear_frame = extract_red_line(rear_frame)
    cleaned_rear_frame = smooth_image(remove_small_connected_areas(redline_rear_frame))
    edge_detected_rear_frame = edge_detection(cleaned_rear_frame)
    #coordinates = get_white_pixel_coordinates(edge_detected_rear_frame)
    #organized_coordinates = organize_datapoints_by_horizontal_coordinate(coordinates)
    #plt.figure(1)
    #plt.scatter(coordinates[:,1], coordinates[:,0], s=1, c='r', marker="s", label='redline')
    #plt.plot(organized_coordinates[:,1], organized_coordinates[:,0], c='m', label='redline')
    #organized_coordinates[:,0] = organized_coordinates[:,0] - organized_coordinates[0,0]
    #integrated_coordinates = integrate_datapoints(organized_coordinates)
    #plt.plot(integrated_coordinates[:,1], integrated_coordinates[:,0], c='y', label='redline')
    #plt.show(block=False)
    #input('Press Enter to continue...')
    inverted_sat_rear_frame = invert_image(saturation_rear_frame)
    inverted_green_rear_frame = invert_image(green_rear_frame)
    inverted_blue_rear_frame = invert_image(blue_rear_frame)
    line_red_rear_frame_indices = get_highest_intensity_pixel_indices(red_rear_frame)
    line_value_rear_frame_indices = get_highest_intensity_pixel_indices(value_rear_frame)
    line_inverted_sat_rear_frame_indices = get_highest_intensity_pixel_indices(inverted_sat_rear_frame)
    line_blue_rear_frame_indices = get_highest_intensity_pixel_indices(inverted_blue_rear_frame)
    line_green_rear_frame_indices = get_highest_intensity_pixel_indices(inverted_green_rear_frame)
    line_hue_rear_frame_indices = get_highest_intensity_pixel_indices(hue_rear_frame)
    average_rear_frame_indices = np.zeros((line_red_rear_frame_indices.shape[0], 2))
    average_rear_frame_indices[:,0] = (line_red_rear_frame_indices[:,0] + line_value_rear_frame_indices[:,0] + line_inverted_sat_rear_frame_indices[:,0] + line_blue_rear_frame_indices[:,0] + line_green_rear_frame_indices[:,0] + line_hue_rear_frame_indices[:,0])/6
    average_rear_frame_indices[:,1] = (line_red_rear_frame_indices[:,1])
    plt.figure(2)
    plt.plot(average_rear_frame_indices[:,1], average_rear_frame_indices[:,0], c='m')
    plt.xlim(0, rear_frame.shape[1])
    plt.ylim(0, rear_frame.shape[0])
    plt.show(block=False)
    line_red_rear_frame = rear_frame+0
    line_value_rear_frame = line_red_rear_frame+0
    line_inverted_sat_rear_frame = line_value_rear_frame+0
    line_hue_rear_frame = line_inverted_sat_rear_frame+0
    line_blue_rear_frame = line_hue_rear_frame+0
    line_green_rear_frame = line_blue_rear_frame+0
    line_red_rear_frame[line_red_rear_frame_indices[:,0], line_red_rear_frame_indices[:,1]] = [0,255,255]
    line_value_rear_frame[line_value_rear_frame_indices[:,0], line_value_rear_frame_indices[:,1]] = [0,255,255]
    line_inverted_sat_rear_frame[line_inverted_sat_rear_frame_indices[:,0], line_inverted_sat_rear_frame_indices[:,1]] = [0,255,255]
    line_hue_rear_frame[line_hue_rear_frame_indices[:,0], line_hue_rear_frame_indices[:,1]] = [0,255,255]
    line_blue_rear_frame[line_blue_rear_frame_indices[:,0], line_blue_rear_frame_indices[:,1]] = [0,255,255]
    line_green_rear_frame[line_green_rear_frame_indices[:,0], line_green_rear_frame_indices[:,1]] = [0,255,255]
    cv2.imshow('rear_red', make_image_bigger(line_red_rear_frame,1000))
    cv2.imshow('rear_hue', make_image_bigger(line_hue_rear_frame,1000))
    cv2.imshow('rear_value', make_image_bigger(line_value_rear_frame,1000))
    cv2.imshow('rear_sat', make_image_bigger(line_inverted_sat_rear_frame,1000))
    cv2.imshow('rear_blue', make_image_bigger(line_blue_rear_frame,1000))
    cv2.imshow('rear_green', make_image_bigger(line_green_rear_frame,1000))
    cv2.waitKey(1)
    #plt.close()
    #print(i)
    #rear_area[i] = integrated_coordinates[-1,0]
    input('Press Enter to continue...')
    plt.close()

time = np.linspace(0, front_num_frames, front_num_frames)/frame_rate

plt.figure(1)
plt.plot(time, front_area, c='r', label='front')
plt.figure(2)
plt.plot(time, rear_area, c='b', label='rear')
plt.figure(3)
plt.plot(time, rear_area-front_area, c='r', label='front')
plt.show(block=False)
input('Press Enter to continue...')

cv2.destroyAllWindows()