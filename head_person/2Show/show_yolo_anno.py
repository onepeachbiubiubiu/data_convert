#-*-coding:utf-8-*-
# date:2019-08
# Author: Eric.Lee
# function: show yolo datasets anno

import cv2
import os
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--anno", required=True,help="path to the labels of images")
ap.add_argument("-i", "--image", required=True,help="path to the labels of images")
ap.add_argument("-o", "--out", required=True,help="path to output images")
args = vars(ap.parse_args())
#read the arguments
anno_path = args["anno"]
img_path = args["image"]
out_path = args["out"]
print("ANNO_PATH : "+anno_path+"\n")
print("IMAGE_PATH : "+img_path+"\n")
print("OUT_PATH : "+out_path+"\n")
os.makedirs(out_path,exist_ok=True)
txts = os.listdir(anno_path)
for i in range(len(txts)):
    # print(txts[i])
    name = txts[i].replace(".txt", ".jpg")
    img = cv2.imread(os.path.join(img_path,name))
    w = img.shape[1]
    h = img.shape[0]

    label_path = os.path.join(anno_path,txts[i])
    # print(i,label_path)
    if os.path.isfile(label_path):
        with open(label_path, 'r') as file:
            lines = file.read().splitlines()

        x = np.array([x.split() for x in lines], dtype=np.float32)
    for k in range(len(x)):
        anno = x[k]
        label = int(anno[0])
        x1 = int((float(anno[1])-float(anno[3])/2)*w)
        y1 = int((float(anno[2])-float(anno[4])/2)*h)

        x2 = int((float(anno[1])+float(anno[3])/2)*w)
        y2 = int((float(anno[2])+float(anno[4])/2)*h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (255,30,30), 2)

    cv2.imwrite(os.path.join(out_path,name),img)

