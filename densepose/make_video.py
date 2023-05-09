import cv2
import numpy as np
import os
import subprocess
from subprocess import PIPE, run,Popen

image_folder = '.'
video_name = 'zlatan_output1.avi'

def sort_fun(img):
    x=img.split(".")[0]
    x=int(x)
    return x

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=sort_fun)
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()