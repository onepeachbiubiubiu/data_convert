import os
import cv2
import json
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("-a",
                "--anno",
                required=True,
                help="path to the labels of images")
ap.add_argument("-i", "--image", required=True, help="path to images")
ap.add_argument("-o", "--out", required=True, help="path to output images")
ap.add_argument("-r", "--ratio", required=True, help="enlarge ratio")
args = vars(ap.parse_args())
#read the arguments
anno_path = args["anno"]
out_path = args["out"]
img_path = args["image"]
ratio = eval(args["ratio"])
# originLabelsDir = "/mnt/data1/lihan/YOLOX/datasets/crowdhuman/labels"
# originImagesDir = "/mnt/data1/lihan/YOLOX/datasets/crowdhuman/train2017"
os.makedirs(os.path.join(out_path, "annotations"), exist_ok=True)
new_img_path = os.path.join(out_path, "train2017")
out_path = os.path.join(out_path, "annotations/instances_train2017.json")
json_name = out_path
dataset = {
    'categories': [{
        'id': 1,
        'name': "person",
        'supercategory': "0"
    }],
    'annotations': [],
    'images': []
}
img_count, box_count = 0, 0
frame_box_num = 0
print("ANNO_PATH : " + anno_path + "\n")
print("IMAGE_PATH : " + img_path + "\n")
print("OUT_PATH : " + out_path + "\n")

txts = os.listdir(anno_path)
# import pdb;pdb.set_trace()
for t in range(len(txts)):
    # fozr t in range(3):
    # print(txts[i])
    if not txts[t].endswith("txt"):
        continue
    video = txts[t].replace(".txt", "")
    video_dir = os.path.join(img_path, video)
    if not os.path.exists(video_dir):
        continue
    label_path = os.path.join(anno_path, txts[t])
    imgs = os.listdir(video_dir)
    if os.path.isfile(label_path):
        with open(label_path, 'r') as file:
            lines = file.read().splitlines()
    else:
        print("Something impossible about label")

    nowframe = "0000x"
    img = None
    line_num = len(lines)
    video_frame = 0
    # box_frame = []

    for i in tqdm(range(line_num)):
        # print(line)
        line = lines[i]
        eles = line.split(",")
        if eles[0] != nowframe:
            nowframe = eles[0]
            video_frame += 1
            if video_frame % 5 != 0:
                img_count += 1
                image_name = video + "_" + nowframe + ".jpg"
                if image_name not in imgs:
                    print("Something impossible about image")
                # img = cv2.imread(os.path.join(video_dir,image_name))
                # H, W, _ = img.shape
                #resize
                image = Image.open(os.path.join(video_dir, image_name))
                W, H = image.size
                W = int(W * ratio)
                H = int(H * ratio)
                image = image.resize((W, H), Image.BICUBIC)
                # import pdb;pdb.set_trace()
                os.makedirs(os.path.join(new_img_path, video), exist_ok=True)
                image.save(os.path.join(new_img_path, video, image_name))
                #
                dataset['images'].append({
                    'file_name':
                    os.path.join(video, image_name),
                    'id':
                    img_count,
                    'width':
                    W,
                    'height':
                    H
                })
                # box_count -= frame_box_num
                # frame_box_num = 0
            else:
                # import pdb;pdb.set_trace()
                continue
                # dataset['annotations'].extend(box_frame)
                # frame_box_num = 0
            # box_frame = []
        else:
            if video_frame % 5 == 0:
                continue
        clss = int(eles[1])
        if clss != 0:
            continue
        # import pdb;pdb.set_trace()
        # if eles[2] == "":
        #     continue
        # if eles[2] == "3.":
        #     clss_num = 3
        # else:
        #     clss_num = int(eles[2])
        x1 = int(int(eles[3]) * ratio)
        y1 = int(int(eles[4]) * ratio)
        x2 = int(int(eles[5]) * ratio)
        y2 = int(int(eles[6]) * ratio)
        box_count += 1
        # frame_box_num +=1
        width = x2 - x1
        height = y2 - y1
        dataset['annotations'].append({
            'area':
            width * height,
            # COCO format
            'bbox': [x1, y1, width, height],
            'category_id':
            1,
            'id':
            box_count,
            'image_id':
            img_count,
            'iscrowd':
            0,
            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
        })
    print("frame: {}, box: {}".format(img_count, box_count))
    print(t, "/", len(txts))

with open(json_name, 'w') as f:
    json.dump(dataset, f)