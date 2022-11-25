#-*-coding:utf-8-*-
# date:2019-08
# Author: Eric.Lee
# function: show yolo datasets anno

import cv2
import os
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-a",
                "--anno",
                required=True,
                help="path to the labels of images")
# ap.add_argument("-i", "--image", required=True,help="path to the labels of images")
ap.add_argument("-o", "--out", required=True, help="path to output images")
args = vars(ap.parse_args())
#read the arguments
anno_path = args["anno"]
# img_path = args["image"]
out_path = args["out"]
print("ANNO_PATH : " + anno_path + "\n")
# print("IMAGE_PATH : "+img_path+"\n")
print("OUT_PATH : " + out_path + "\n")
os.makedirs(out_path, exist_ok=True)
txts = os.listdir(anno_path)
# import pdb;pdb.set_trace()
for i in range(len(txts)):
    # print(txts[i])
    if not txts[i].endswith("txt"):
        continue
    video = txts[i].replace(".txt", "")
    video_dir = os.path.join(anno_path, video)
    if not os.path.exists(video_dir):
        continue
    label_path = os.path.join(anno_path, txts[i])
    imgs = os.listdir(video_dir)
    if os.path.isfile(label_path):
        with open(label_path, 'r') as file:
            lines = file.read().splitlines()
    else:
        print("Something impossible about label")
    nowframe = "00000"
    img = None
    for line in lines:
        eles = line.split(",")
        if eles[0] != nowframe:
            if nowframe != "00000":
                os.makedirs(os.path.join(out_path, video), exist_ok=True)
                cv2.imwrite(
                    os.path.join(out_path, video,
                                 video + "_" + nowframe + ".jpg"), img)
            nowframe = eles[0]
            image_name = video + "_" + nowframe + ".jpg"
            if image_name not in imgs:
                print("Something impossible about image")
            img = cv2.imread(os.path.join(video_dir, image_name))
        clss = int(eles[1])
        if clss != 0:
            continue
        # import pdb;pdb.set_trace()
        # clss_num = eles[2]
        # if  "." in clss_num or "]" in clss_num:
        #     print(line)
        #     x1 = int(eles[3])
        #     y1 = int(eles[4])
        #     x2 = int(eles[5])
        #     y2 = int(eles[6])
        #     img = cv2.imread(os.path.join(video_dir,image_name))
        #     cv2.rectangle(img, (x1,y1), (x2,y2), (255,30,30), 2)
        #     os.makedirs(os.path.join(out_path,video),exist_ok=True)

        #     cv2.imwrite(os.path.join(out_path,video,video+"_"+nowframe+".jpg"),img)
        #     print(os.path.join(out_path,video,video+"_"+nowframe+".jpg"))
        #     nowframe = "00000"
        #     break
        # if(eles[2]==""):
        #     continue
        # clss_num = int(eles[2])
        x1 = int(eles[3])
        y1 = int(eles[4])
        x2 = int(eles[5])
        y2 = int(eles[6])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 30, 30), 2)
