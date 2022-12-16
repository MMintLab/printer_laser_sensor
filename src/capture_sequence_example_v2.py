from time import sleep
from picamera import PiCamera
import time
from PIL import Image
import PIL.ImageOps as imops
import io

camera = PiCamera(resolution=(640, 480), framerate=80)
# Wait for the automatic gain control to settle
camera.shutter_speed = 5000

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
        img = Image.open(stream)
        print('inverted')
        img2 = imops.invert(img)
        img2.save('image%d.jpg' % i)
        stream.seek(0)
        stream.truncate()

# Now fix the values
# Finally, take several photos with the fixed settings
starttime = time.time()
camera.capture_sequence(outputs(10),use_video_port=True,burst=False)
print(time.time()-starttime)
camera.close()