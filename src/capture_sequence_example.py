from time import sleep
from picamera import PiCamera
import time

camera = PiCamera(resolution=(640, 480), framerate=80)
# Wait for the automatic gain control to settle
sleep(2)
# Now fix the values
# Finally, take several photos with the fixed settings
starttime = time.time()
camera.capture_sequence(['image%d.jpg' % i for i in range(10)])
print(time.time()-starttime)
camera.close()