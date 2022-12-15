import cv2
import sys
import os
import numpy as np

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

#focus on lower part of image based on given pixel count
def focus_on_lower_part(image, pixel_count):
    height, width, channels = image.shape
    image = image[height-pixel_count:height, 0:width]
    return image

#edge detection
def edge_detection(image):
    edges = cv2.Canny(image,100,200)
    return edges

#make image grayscale
def make_image_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


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
for i in range(front_num_frames-624):
    front_frame_path = str(front_frame_folder)+"/front_frame"+str(i)+".jpeg"
    front_frame = focus_on_lower_part_percentage(cv2.imread(front_frame_path),0.51)
    hsv_front_frame = convert_image_to_hsv(front_frame)
    redline_front_frame = extract_red_line(front_frame)
    cleaned_front_frame = smooth_image(remove_small_connected_areas(redline_front_frame))
    cv2.imshow('front_frame', sobel_edge_detection(cleaned_front_frame))
    print(i)
    cv2.waitKey(0)

#iterate through the rear frames only and show each of them with a pause in between
for i in range(rear_num_frames):
    rear_frame_path = str(rear_frame_folder)+"/rear_frame"+str(i)+".jpeg"
    rear_frame = focus_on_lower_part_percentage(cv2.imread(rear_frame_path),0.55)
    hsv_rear_frame = convert_image_to_hsv(rear_frame)
    redline_rear_frame = extract_red_line(rear_frame)
    cleaned_rear_frame = smooth_image(remove_small_connected_areas(redline_rear_frame))
    cv2.imshow('rear_frame', sobel_edge_detection(cleaned_rear_frame))
    print(i)
    cv2.waitKey(0)

cv2.destroyAllWindows()