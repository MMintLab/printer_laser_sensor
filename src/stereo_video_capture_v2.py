import cv2
import os
import numpy as np
from picamera import PiCamera
import time
import RPi.GPIO as GPIO
from smbus import SMBus
import io

#set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

duration = 60 #seconds
frame_rate = 30 #fps
end_frame = frame_rate * duration

time_between_frames = 1/frame_rate

previous_time = time.time()

def switch_camera():
    if GPIO.input(4):
        SMBus(1).write_byte_data(0x70, 0x00, 0x01)
        GPIO.output(4, GPIO.LOW)
        GPIO.output(17, GPIO.LOW)
    else:
        SMBus(1).write_byte_data(0x70, 0x00, 0x02)
        GPIO.output(4, GPIO.HIGH)
        GPIO.output(17, GPIO.LOW)

#take picture with raspberry pi camera
def take_picture(camera,frame_number,camera_name):
    camera.resolution = (1280, 720)
    camera.capture(camera_name+'_frame'+str(frame_number)+'.jpg')

def outputs():
    stream = io.BytesIO()
    yield stream
    switch_camera()
    stream.seek(0)
    stream.truncate()

time.sleep(2)

with PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 80
    time.sleep(2)
    start = time.time()
    camera.capture_sequence(outputs(), 'jpeg', use_video_port=True)
    finish = time.time()
    print('Captured 40 frames at %.2ffps' % (40 / (finish - start)))
    camera.close()

GPIO.cleanup()
print("Done")
