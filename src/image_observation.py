import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

front_frame_folder = '/Users/william/Documents/MMINT_Research/printer_laser_sensor/sample_videos/front_frames'
rear_frame_folder = '/Users/william/Documents/MMINT_Research/printer_laser_sensor/sample_videos/rear_frames'

frame_rate = 29.97

def get_num_frames(folder):
    num_frames = 0
    for file in os.listdir(folder):
        if file.endswith(".jpeg"):
            num_frames += 1
    return num_frames

def convert_image_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

front_num_frames = get_num_frames(front_frame_folder)

rear_num_frames = get_num_frames(rear_frame_folder)

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

#iterate through the front frames only and show each of them with a pause in between
front_area = np.zeros((front_num_frames, 1))
for i in range(front_num_frames):
    front_frame_path = str(front_frame_folder)+"/front_frame"+str(i)+".jpeg"
    front_frame = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(cv2.imread(front_frame_path),0.51), 0.5),0.5)
    hsv_front_frame = convert_image_to_hsv(front_frame)
    redline_front_frame = extract_red_line(front_frame)
    cleaned_front_frame = smooth_image(remove_small_connected_areas(redline_front_frame))
    edge_detected_front_frame = edge_detection(cleaned_front_frame)
    coordinates = get_white_pixel_coordinates(edge_detected_front_frame)
    organized_coordinates = organize_datapoints_by_horizontal_coordinate(coordinates)
    #plt.figure(1)
    #plt.scatter(coordinates[:,1], coordinates[:,0], s=1, c='r', marker="s", label='redline')
    #plt.plot(organized_coordinates[:,1], organized_coordinates[:,0], c='m', label='redline')
    organized_coordinates[:,0] = organized_coordinates[:,0] - organized_coordinates[0,0]
    integrated_coordinates = integrate_datapoints(organized_coordinates)
    #plt.plot(integrated_coordinates[:,1], integrated_coordinates[:,0], c='y', label='redline')
    #plt.show(block=False)
    #input('Press Enter to continue...')
    #cv2.imshow('front_frame', edge_detected_front_frame)
    #plt.close()
    print(i)
    front_area[i] = integrated_coordinates[-1,0]
    #cv2.waitKey(0)

#iterate through the rear frames only and show each of them with a pause in between
rear_area = np.zeros((rear_num_frames, 1))
for i in range(rear_num_frames):
    rear_frame_path = str(rear_frame_folder)+"/rear_frame"+str(i)+".jpeg"
    rear_frame = focus_on_upper_part_percentage(crop_image_horizontally(focus_on_lower_part_percentage(cv2.imread(rear_frame_path),0.55), 0.5),0.5)
    hsv_rear_frame = convert_image_to_hsv(rear_frame)
    redline_rear_frame = extract_red_line(rear_frame)
    cleaned_rear_frame = smooth_image(remove_small_connected_areas(redline_rear_frame))
    edge_detected_rear_frame = edge_detection(cleaned_rear_frame)
    coordinates = get_white_pixel_coordinates(edge_detected_rear_frame)
    organized_coordinates = organize_datapoints_by_horizontal_coordinate(coordinates)
    organized_coordinates[:,0] = organized_coordinates[:,0] - organized_coordinates[0,0]
    #plt.figure(2)
    #plt.scatter(coordinates[:,1], coordinates[:,0], s=1, c='b', marker="s", label='redline')
    #plt.plot(organized_coordinates[:,1], organized_coordinates[:,0], c='c', label='redline')
    integrated_coordinates = integrate_datapoints(organized_coordinates)
    #plt.plot(integrated_coordinates[:,1], integrated_coordinates[:,0], c='g', label='redline')
    #plt.show(block=False)
    #input('Press Enter to continue...')
    #close figure
    #plt.close()
    #cv2.imshow('rear_frame', edge_detected_rear_frame)
    print(i)
    rear_area[i] = integrated_coordinates[-1,0]
    #cv2.waitKey(0)

cv2.destroyAllWindows()