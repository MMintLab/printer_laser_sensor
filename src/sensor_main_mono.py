from time import sleep
from picamera import PiCamera
import time
from PIL import Image
import PIL.ImageOps as imops
import io
import cv2
import numpy as np
import RPi.GPIO as GPIO
from smbus import SMBus
import rospy
import sys
from std_msgs.msg import Float32
#standard messages uint8 image import
from sensor_msgs.msg import Image

pub_im = rospy.Publisher('/camera_image', Image, queue_size=1)
rospy.init_node('sensor_node')

def laser_seg(R,G,B,S,V,L):
    #weights = np.array([[2.77323259],[-4.28669185],[2.87523104],[-0.40948264],[-2.14548059],[0.69278767],[1.30010453]])
    #divided by 255: weights = np.array([[0.010875421921568628],[-0.016810556274509805],[0.011275415843137255],[-0.0016058142745098039],[-0.00841364937254902],[1.0654174086889657e-05],[1.9993918185313344e-05]])
    #integer weights multiplied by 1000000
    weights = np.array([[10875],[-16811],[11275],[-1606],[-8414],[11],[20]])
    output = L*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B + weights[3,0]*S + weights[4,0]*V + weights[5,0]*S*V + weights[6,0]*S*L)
    #clip output to be between 0 and 1
    output[output>255000000] = 255000000
    output[output<0] = 0
    max_val = np.max(output)
    min_val = np.min(output)
    middle = 1*(max_val+min_val)/4
    #output[output>middle] = 1
    output[output<=middle] = 0
    return output

def nozzle_seg(R,G,B,S):
    #weights = np.array([[-1.18942075],[2.44500537],[-1.3129521],[4.140930960918501],[-2.885439160129449]])
    weights = np.array([[-4664],[9588],[-5149],[4141],[-11]])
    output = weights[3,0]*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B)+weights[4,0]*S*(weights[0,0]*R + weights[1,0]*G + weights[2,0]*B)
    #clip output to be between 0 and 1
    output[output>1000000000] = 1000000000
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

    green_channel = image[:,:,1]

    blue_channel = image[:,:,0]

    sat_channel = hsl_image[:,:,2]

    value_channel = hsv_image[:,:,2]

    lum_channel = hsl_image[:,:,1]

    return red_channel, green_channel, blue_channel, sat_channel, value_channel, lum_channel


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

capture_framerate = 5

desired_camera_arg = int(sys.argv[1])

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

i2cbus = SMBus(1)

def switch_camera(desired_camera,i2cbus):
    if desired_camera==1:
        i2cbus.write_byte_data(0x70, 0x00, 0x02)
        GPIO.output(4, GPIO.HIGH)
        GPIO.output(17, GPIO.LOW)
    elif desired_camera==0:
        i2cbus.write_byte_data(0x70, 0x00, 0x01)
        GPIO.output(4, GPIO.LOW)
        GPIO.output(17, GPIO.LOW)

camera = PiCamera(resolution=(200, 200), framerate=capture_framerate)
# Wait for the automatic gain control to settle

sleep(2)
camera.shutter_speed = 50000
camera.zoom = (0.3,0.3,0.4,0.4)
camera.iso = 800

#display camera fields
print(camera.resolution)
print(camera.framerate)
print(camera.exposure_speed)
print(camera.iso)
print(camera.awb_gains)
print(camera.awb_mode)
print(camera.shutter_speed)
print(camera.exposure_mode)

switch_camera(desired_camera_arg,i2cbus)

def outputs(i2cbus,starttime):
    running_average_number = 10
    stream = io.BytesIO()
    while True:
        yield stream
        #get image from stream in opencv format
        imagetime = time.time()-starttime
        data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, 1)
        #get channels
        #red_channel, green_channel, blue_channel, sat_channel, value_channel, lum_channel = get_channels(image)
        #laser_seg_image = laser_seg(red_channel,green_channel,blue_channel,sat_channel,value_channel,lum_channel)
        #nozzle_seg_image = nozzle_seg(red_channel,green_channel,blue_channel,sat_channel)
        #cv2.imshow('nozzle_seg',nozzle_seg_image)
        #cv2.imshow('laser_seg',laser_seg_image)
        #get vertical coordinate of highest white pixel in nozzle_seg_image
        #nozzle_column_sum = np.sum(nozzle_seg_image,axis=1)
        #get lowest index of nonzero element in nozzle_column_sum
        #nozzle_index = np.nonzero(nozzle_column_sum)[0][0]
        #print(nozzle_index)
        #crop laser_seg_image to only include pixels above nozzle
        #laser_seg_image = laser_seg_image[0:nozzle_index,:]
        #show image
        print(image.shape)
        cv2.imshow('vision',image)
        cv2.waitKey(1)
        #get highest intensity pixel in each column of laser_seg_image
        #highest_intensity_pixel_indices = get_highest_intensity_pixel_indices(laser_seg_image)
        #get average of last 4 highest intensity pixel indices
        #y2 = np.mean(highest_intensity_pixel_indices[0,-4:])
        #get average of first 4 highest intensity pixel indices
        #y1 = np.mean(highest_intensity_pixel_indices[0,0:4])
        #get average of last 4 column numbers
        #x2 = np.mean(np.linspace(start=image.shape[1]-4, stop=image.shape[1]-1, num=4, axis=0))
        #get average of first 4 column numbers
        #x1 = np.mean(np.linspace(start=0, stop=3, num=4, axis=0))
        #get slope of line connecting (x1,y1) and (x2,y2)
        #slope = (y2-y1)/(x2-x1)
        #subtract slope times column number from highest intensity pixel indices
        #highest_intensity_pixel_indices = highest_intensity_pixel_indices - slope*np.linspace(start=0, stop=image.shape[1]-1, num=image.shape[1], axis=0)
        #running average of highest intensity pixel indices
        #highest_intensity_pixel_indices = running_average(highest_intensity_pixel_indices,N=running_average_number)
        #highest_intensity_pixel_indices = highest_intensity_pixel_indices - np.min(highest_intensity_pixel_indices)
        #bead_area = np.sum(highest_intensity_pixel_indices)
        #print(bead_area)
        pub_im.publish(image)
        stream.seek(0)
        stream.truncate()

# Now fix the values
# Finally, take several photos with the fixed settings
starttime = time.time()
camera.capture_sequence(outputs(i2cbus,starttime),use_video_port=True,burst=False)
print(time.time()-starttime)
camera.close()
GPIO.cleanup()
