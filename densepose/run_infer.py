import cv2
import numpy as np
import os
import subprocess
from subprocess import PIPE, run, Popen

def predict1(img):
    outf = img + "_output" + ".png"
    folder = "./coco_test_images/"

    in_file=os.path.join(folder,img)
    out_file = outf

    t = subprocess.run(["python3", "apply_net.py", "show", "configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml",
                       "densepose/models/model_final_d366fa.pkl", in_file, "bbox,dp_segm", "--output", out_file], capture_output=False)

    return out_file

image_folder = "coco_test_images/"
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
for img in images:
    predict1(img)
