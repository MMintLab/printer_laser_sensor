import cv2
import sys
import os
import numpy as np

front_video_path = 'printer_laser_sensor/sample_videos/front.mp4'
rear_video_path = 'printer_laser_sensor/sample_videos/rear.mp4'

front_frame_folder = 'printer_laser_sensor/sample_videos/front_frames'
rear_frame_folder = 'printer_laser_sensor/sample_videos/rear_frames'

frontcap = cv2.VideoCapture(front_video_path)
frontsuccess, frontimage = frontcap.read()
frontcount = 0

#print(str(front_frame_folder)+"/frame"+str(frontcount)+".jpeg")
while frontsuccess:
    cv2.imwrite(str(front_frame_folder)+"/front_frame"+str(frontcount)+".jpeg", frontimage)
    #print(frontcount)
    frontsuccess, frontimage = frontcap.read()
    frontcount = frontcount + 1

frontcap.release()

rearcap = cv2.VideoCapture(rear_video_path)
rearsuccess, rearimage = rearcap.read()
rearcount = 0

while rearsuccess:
    cv2.imwrite(str(rear_frame_folder)+"/rear_frame"+str(rearcount)+".jpeg", rearimage)
    rearsuccess, rearimage = rearcap.read()
    rearcount = rearcount + 1

rearcap.release()

print("frames have been extracted")

