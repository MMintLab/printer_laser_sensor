from time import sleep
from picamera import PiCamera
import time
from PIL import Image
import PIL.ImageOps as imops
import io
import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
from smbus import SMBus

#clear files to write to
f0 = open('camera00time.txt','w')
f1 = open('camera01time.txt','w')
f0.close()
f1.close()

#open file to write to
f0 = open('camera00time.txt','a')
f1 = open('camera01time.txt','a')

#store time values in file
def store_time(camera_number,imagetime):
    if camera_number==0:
        f0.write(str(imagetime)+'\n')
    elif camera_number==1:
        f1.write(str(imagetime)+'\n')


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

camera = PiCamera(resolution=(100, 100), framerate=20)
# Wait for the automatic gain control to settle

sleep(2)
camera.shutter_speed = 10000

#display camera fields
print(camera.resolution)
print(camera.framerate)
print(camera.exposure_speed)
print(camera.iso)
print(camera.awb_gains)
print(camera.awb_mode)
print(camera.shutter_speed)
print(camera.exposure_mode)

def outputs(numphotos,i2cbus,starttime):
    stream = io.BytesIO()
    for i in range(numphotos):
        yield stream
        switch_camera(i%2,i2cbus)
        #get image from stream in opencv format
        imagetime = time.time()-starttime
        data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        img = cv.imdecode(data, 1)
        cv.imwrite('camera%dimage%02d.jpg' % (i%2,int(i/2)), img)
        store_time(i%2,imagetime)
        stream.seek(0)
        stream.truncate()

# Now fix the values
# Finally, take several photos with the fixed settings
starttime = time.time()
camera.capture_sequence(outputs(100,i2cbus,starttime),use_video_port=True,burst=False)
print(time.time()-starttime)
camera.close()
GPIO.cleanup()
f1.close()
f0.close()