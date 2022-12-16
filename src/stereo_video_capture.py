import cv2
import os
import numpy as np
from picamera import PiCamera
import time
import RPi.GPIO as GPIO
from smbus import SMBus

#set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

i2cbus = SMBus(1)

duration = 60 #seconds
frame_rate = 30 #fps
end_frame = frame_rate * duration

time_between_frames = 1/frame_rate

previous_time = time.time()

def switch_camera(desired_camera,i2cbus):
    if desired_camera=='front':
        i2cbus.write_byte_data(0x70, 0x00, 0x02)
        GPIO.output(4, GPIO.HIGH)
        GPIO.output(17, GPIO.LOW)
    elif desired_camera=='rear':
        i2cbus.write_byte_data(0x70, 0x00, 0x01)
        GPIO.output(4, GPIO.LOW)
        GPIO.output(17, GPIO.LOW)

#take picture with raspberry pi camera
def take_picture(camera,frame_number,camera_name):
    camera.capture(camera_name+'_frame'+str(frame_number)+'.jpg')

time.sleep(2)

camera = PiCamera()
camera.resolution = (1280, 720)

try:
    for frame_number in range(0,end_frame):
        current_time = time.time()
        if current_time - previous_time > time_between_frames:
            previous_time = current_time
            switch_camera('front',i2cbus)
            take_picture(camera,frame_number,'front')
            switch_camera('rear',i2cbus)
            take_picture(camera,frame_number,'rear')
            print(frame_number)
except:
    camera.close()
    GPIO.cleanup()
    print('Failed')


camera.close()
GPIO.cleanup()
print("Done")
