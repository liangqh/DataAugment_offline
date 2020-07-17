import cv2
import torch
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
#import xml.dom.minidom
import shutil
import glob

img_path = '/media/cjh/code/data/img_test'
imgnames = os.listdir(img_path)
label_path = '/media/cjh/code/data/label_test'
labelnames = os.listdir(label_path)

imgnames_hou = []
for s in imgnames:
    imgnames_hou.append(s.split(".")[0])

labelnames_hou = []
for l in labelnames:
    labelnames_hou.append(l.split(".")[0])

for label in labelnames_hou:
    if label not in imgnames_hou:
        path = label_path + '/' + label + '.xml'
        print("path:", path)
        os.remove(path)

