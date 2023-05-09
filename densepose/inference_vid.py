import cv2
import numpy as np
import os
import subprocess
from subprocess import PIPE, run,Popen

captura = cv2.VideoCapture('zlatan1.mp4')
    
def predict1(img,no):
    outf=str(no)+".png"
    out_file=outf
    t=subprocess.run(["python3","apply_net.py","show","configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml","densepose/models/model_final_d366fa.pkl",str(img),"dp_contour,bbox","--output",out_file],capture_output=False)
    return out_file
currentFrame=0
    
while(1):
    ret, frame = captura.read()
    print ('Creating...' + name)
    print(name)
    cv2.imwrite(name,frame)
    predict1(name, currentFrame)

    currentFrame += 1

captura.release()
cv2.destroyAllWindows()

image_folder = ''

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images=sorted(images)

image_folder = '.'
video_name = 'zlatan1_out_video.mp4'

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

for file in os.listdir('.'):
    if file.endswith('.png'):
        os.remove(file)
    if file.endswith('.jpg'):
        os.remove(file)




