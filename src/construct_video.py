import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

#first argument from terminal is the folder containing the frames
image_folder = sys.argv[1]
#second argument from terminal is the test number
testnumber = int(sys.argv[2])
#third argument from terminal is the camera number
cameranumber = int(sys.argv[3])

#use cv2 to save set of images as mp4 video given test number and camera number
def save_video_mp4(image_folder, testnumber, cameranumber):
    video_name = 'test%02dcamera%d.mp4' % (testnumber, cameranumber)
    images = [img for img in os.listdir(image_folder) if img.startswith(str('test%02dcamera%dimage' % (testnumber, cameranumber)))]
    #sort images by number i.e. test01camera1image00001.jpeg, test01camera1image00002.jpeg, etc.
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()

save_video_mp4(image_folder, testnumber, cameranumber)
