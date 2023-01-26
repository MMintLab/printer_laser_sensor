import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import time

data_folder = 'data/Jan11_2023'

#test_number = int(sys.argv[1])

def laser_seg(R,G,B,S,V,L):
    weights = np.array([[2.77323259],[-4.28669185],[2.87523104],[-0.40948264],[-2.14548059],[0.69278767],[1.30010453]])
    output = L*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B + weights[3,0]*S + weights[4,0]*V + weights[5,0]*S*V + weights[6,0]*S*L)
    #clip output to be between 0 and 1
    output[output>1] = 1
    output[output<0] = 0
    max_val = np.max(output)
    min_val = np.min(output)
    middle = 1*(max_val+min_val)/4
    #output[output>middle] = 1
    output[output<=middle] = 0
    return output

def nozzle_seg(R,G,B,S):
    weights = np.array([[-1.18942075],[2.44500537],[-1.3129521],[4.140930960918501],[-2.885439160129449]])
    output = weights[3,0]*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B)+weights[4,0]*S*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B)
    #clip output to be between 0 and 1
    output[output>1] = 1
    output[output<0] = 0
    max_val = np.max(output)
    min_val = np.min(output)
    middle = 3*(max_val+min_val)/4
    output[output>middle] = 1
    output[output<=middle] = 0
    return output

#return red, green, blue, saturation, value, and luminance channels of image
def get_channels(image):
    #convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #get red, green, blue, saturation, value, and luminance channels
    red_channel = image[:,:,2]
    red_channel = red_channel/255

    green_channel = image[:,:,1]
    green_channel = green_channel/255

    blue_channel = image[:,:,0]
    blue_channel = blue_channel/255

    sat_channel = hsl_image[:,:,2]
    sat_channel = sat_channel/255

    value_channel = hsv_image[:,:,2]
    value_channel = value_channel/255

    lum_channel = hsl_image[:,:,1]
    lum_channel = lum_channel/255

    return red_channel, green_channel, blue_channel, sat_channel, value_channel, lum_channel

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

def get_times(test_number,data_folder):
    filename = 'test%02d_mono_time.txt' % (test_number)
    #load time data as numpy array from file
    time_list = np.loadtxt(data_folder+'/'+filename)
    return time_list

def get_highest_intensity_pixel_indices(image):
    highest_intensity_pixel_indices = np.zeros((1,image.shape[1]))
    for i in range(image.shape[1]):
        if np.sum(image[:,i]) == 0:
            highest_intensity_pixel_indices[0,i] = highest_intensity_pixel_indices[0,i-1]
        else:
            highest_intensity_pixel_indices[0,i] = np.sum((image[:,i]*np.linspace(start=0, stop=image.shape[0]-1, num=image.shape[0], axis=0)))/np.sum(image[:,i])
    return highest_intensity_pixel_indices

def running_average(x, N):
    cumsum = np.cumsum(x, axis=1)
    return (cumsum[:,N:] - cumsum[:,:-N]) / float(N)

test_numbers = [1,2,4,5,6,7,8]
flow_rates = [2,2,2,4,4,4,4]
feed_rates = [5,10,10,5,10,5,10]

start_times = [4,6,4,4,2,8,6]
end_times = [8,12,9,9,7,12,12]

beadCSAperFR = [0,0,0,0,0,0,0]

STDev = [0,0,0,0,0,0,0]

for j in range(len(test_numbers)):
    test_number = test_numbers[j]
    filename_list = get_filename_list(test_number,data_folder)
    time_list = get_times(test_number,data_folder)


    #for all images in filename_list, get red, green, blue, saturation, value, and luminance channels
    #and create laser and nozzle segmentation images
    bead_CSA = np.zeros((np.size(time_list)))
    running_average_number = 10
    for i in range(len(filename_list)):
        image = cv2.imread(data_folder+'/'+filename_list[i])
        red_channel, green_channel, blue_channel, sat_channel, value_channel, lum_channel = get_channels(image)
        laser_seg_image = laser_seg(red_channel,green_channel,blue_channel,sat_channel,value_channel,lum_channel)
        nozzle_seg_image = nozzle_seg(red_channel,green_channel,blue_channel,sat_channel)
        #get vertical coordinate of highest white pixel in nozzle_seg_image
        nozzle_column_sum = np.sum(nozzle_seg_image,axis=1)
        #get lowest index of nonzero element in nozzle_column_sum
        nozzle_index = np.nonzero(nozzle_column_sum)[0][0]
        #print(nozzle_index)
        #crop laser_seg_image to only include pixels above nozzle
        laser_seg_image = laser_seg_image[0:nozzle_index,:]
        #get highest intensity pixel in each column of laser_seg_image
        highest_intensity_pixel_indices = get_highest_intensity_pixel_indices(laser_seg_image)
        #get average of last 4 highest intensity pixel indices
        y2 = np.mean(highest_intensity_pixel_indices[0,-4:])
        #get average of first 4 highest intensity pixel indices
        y1 = np.mean(highest_intensity_pixel_indices[0,0:4])
        #get average of last 4 column numbers
        x2 = np.mean(np.linspace(start=image.shape[1]-4, stop=image.shape[1]-1, num=4, axis=0))
        #get average of first 4 column numbers
        x1 = np.mean(np.linspace(start=0, stop=3, num=4, axis=0))
        #get slope of line connecting (x1,y1) and (x2,y2)
        slope = (y2-y1)/(x2-x1)
        #subtract slope times column number from highest intensity pixel indices
        highest_intensity_pixel_indices = highest_intensity_pixel_indices - slope*np.linspace(start=0, stop=image.shape[1]-1, num=image.shape[1], axis=0)
        #running average of highest intensity pixel indices
        highest_intensity_pixel_indices = running_average(highest_intensity_pixel_indices,N=running_average_number)
        highest_intensity_pixel_indices = highest_intensity_pixel_indices - np.min(highest_intensity_pixel_indices)
        bead_CSA[i] = np.sum(highest_intensity_pixel_indices)
        #get time of image
        #plot highest intensity pixel indices vs column number
        #show laser_seg_image with cv2
        #cv2.imshow('nozzle_seg_image',nozzle_seg_image)
        #cv2.imshow('laser_seg_image',laser_seg_image)
        #plt.figure(figsize=(2,2))
        #plt.plot(np.linspace(start=0, stop=image.shape[1]-1-running_average_number, num=image.shape[1]-running_average_number, axis=0),highest_intensity_pixel_indices[0,:])
        #plt.xlabel('Column Number of Image')
        #plt.ylabel('Height of Laser Line')
        #plt.title('Camera Laser Perception %d' % (i))
        #plt.show()
        #close plot
        #plt.close()
        #cv2.waitKey(1)

    plt.figure(figsize=(5,5))
    #plot bead_CSA vs time, ignoring the first and last 20 values
    plt.plot(time_list[(time_list>start_times[j])&(time_list<end_times[j])],bead_CSA[(time_list>start_times[j])&(time_list<end_times[j])])
    plt.xlabel('Time (s)')
    plt.ylabel('Bead Cross Sectional Area (Square Pixels)')
    plt.title('Bead Cross Sectional Area vs Time'+str(test_number))
    plt.show()
    beadCSAperFR[j] = np.mean(bead_CSA[(time_list>start_times[j])&(time_list<end_times[j])]*feed_rates[j])
    STDev[j] = np.std(bead_CSA[(time_list>start_times[j])&(time_list<end_times[j])]*feed_rates[j])
    cv2.destroyAllWindows()

plt.figure(figsize=(5,5))
#plot beadCSAperFR vs flow_rates including error bars
plt.errorbar(flow_rates,beadCSAperFR,yerr=STDev,fmt='o')
#plot best fit line
z = np.polyfit(flow_rates,beadCSAperFR,1)
p = np.poly1d(z)
plt.plot(flow_rates,p(flow_rates),"r--")
#plot best fit line equation
plt.text(3,600,'y=%.6fx+(%.6f)'%(z[0],z[1]))
plt.xlabel('Flow Rate (mL/min)')
plt.ylabel('Bead Cross Sectional Area Times Feed Rate (Square Pixels*mm/s)')
plt.show()