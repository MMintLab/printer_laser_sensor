import io
import time
import picamera

from PIL import Image

with picamera.PiCamera() as camera:
    # Set the camera's resolution to VGA @40fps and give it a couple
    # of seconds to measure exposure etc.
    camera.resolution = (640, 720)
    camera.framerate = 80
    camera.rotation = 0
    time.sleep(2)
    # Set up 40 in-memory streams
    outputs = [io.BytesIO() for i in range(40)]
    start = time.time()
    print("before capture")
    camera.capture_sequence(outputs, 'jpeg', use_video_port=True)
    print("after capture")
    finish = time.time()
    # How fast were we?
    print('Captured 40 images at %.2ffps' % (40 / (finish - start)))

    count = 0
    for frameData in outputs:
        rawIO = frameData
        rawIO.seek(0)
        byteImg = Image.open(rawIO)

        count += 1
        filename = "testimage" + str(count) + ".jpg"
        byteImg.save(filename, 'JPEG')