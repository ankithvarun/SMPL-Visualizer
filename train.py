from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from src.data.dataset import ParentDataset
from src.data.loader import build_train_loader
from random import randint

COCO_TRAIN_IMAGE_DIR = "/scratch/coco/train2014"
DENSEPOSE_METADATA_DIR = "../metadata"
DENSEPOSE_COCO_ANNOTATIONS_PATH = "../annotations/densepose_coco_2014_train.json"

dataset = ParentDataset("coco_train2014", COCO_TRAIN_IMAGE_DIR, DENSEPOSE_METADATA_DIR, DENSEPOSE_COCO_ANNOTATIONS_PATH, 0.1)
dataset.register()

IMS_PER_BATCH = 16
NUM_WORKERS = 4

loader = build_train_loader(dataset, IMS_PER_BATCH, NUM_WORKERS)

# coco_folder = '/scratch/coco'
# dp_coco = COCO( coco_folder + '/annotations/densepose_coco_2014_train.json')

# Get img id's for the minival dataset.
# im_ids = dp_coco.getImgIds()
# print(len(im_ids))
# # Select a random image id.
# Selected_im = im_ids[randint(0, len(im_ids))] # Choose im no 57 to replicate 
# # Load the image
# im = dp_coco.loadImgs(Selected_im)[0]  
# print(im)
# # Load Anns for the selected image.
# ann_ids = dp_coco.getAnnIds( imgIds=im['id'] )
# anns = dp_coco.loadAnns(ann_ids)
# # Now read and b
# im_name = os.path.join( coco_folder + '/train2014', im['file_name'] )
# I=cv2.imread(im_name)
# plt.imshow(I[:,:,::-1]); plt.axis('off'); plt.show()