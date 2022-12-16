from time import sleep
from picamera import PiCamera

camera = PiCamera(resolution=(1280, 720), framerate=30)

#display camera fields
print(camera.resolution)
print(camera.framerate)
print(camera.exposure_speed)
print(camera.iso)
print(camera.awb_gains)
print(camera.awb_mode)
print(camera.shutter_speed)
print(camera.exposure_mode)

# Set ISO to the desired value
#camera.iso = 100
# Wait for the automatic gain control to settle
sleep(2)
# Finally, take several photos with the fixed settings
camera.capture('image1.jpg')