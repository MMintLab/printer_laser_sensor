from time import sleep
from picamera import PiCamera
import time
from PIL import Image
import PIL.ImageOps as imops
import io
import cv2 as cv
import numpy as np

camera = PiCamera(resolution=(150, 150), framerate=100)
# Wait for the automatic gain control to settle
camera.shutter_speed = 2000

#display camera fields
print(camera.resolution)
print(camera.framerate)
print(camera.exposure_speed)
print(camera.iso)
print(camera.awb_gains)
print(camera.awb_mode)
print(camera.shutter_speed)
print(camera.exposure_mode)

sleep(2)



def outputs(numphotos):
    stream = io.BytesIO()
    for i in range(numphotos):
        yield stream
        #get image from stream in opencv format
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        img = cv.imdecode(data, 1)
        print('inverted')
        #img2 = cv.bitwise_not(img)
        img.save('image%d.jpg' % i)
        stream.seek(0)
        stream.truncate()

# Now fix the values
# Finally, take several photos with the fixed settings
starttime = time.time()
camera.capture_sequence(outputs(10),use_video_port=True,burst=False)
print(time.time()-starttime)
camera.close()