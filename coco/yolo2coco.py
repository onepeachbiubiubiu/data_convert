import os
import cv2
import json
from tqdm import tqdm

originLabelsDir = "~/datasets/crowdhuman/labels"
originImagesDir = "~/datasets/crowdhuman/train2017"
json_name = "~/datasets/crowdhuman/annotations/instances_train2017.json"

dataset = {
    'categories': [{
        'id': 1,
        'name': "0",
        'supercategory': "0"
    }],
    'annotations': [],
    'images': []
}

img_count, box_count = 0, 0

for dataset_name in os.listdir(originImagesDir):
    print(dataset_name)
    imgFileList = os.listdir(os.path.join(originImagesDir, dataset_name))
    imgFileList.sort()
    for imgFile in tqdm(imgFileList):
        imagePath = os.path.join(originImagesDir, dataset_name, imgFile)
        image = cv2.imread(imagePath)
        H, W, _ = image.shape
        img_count += 1
        dataset['images'].append({
            'file_name':
            os.path.join(dataset_name, imgFile),
            'id':
            img_count,
            'width':
            W,
            'height':
            H
        })

        txtFile = imgFile.split('.')[0] + '.txt'
        if os.path.exists(os.path.join(originLabelsDir, dataset_name,
                                       txtFile)):
            with open(os.path.join(originLabelsDir, dataset_name, txtFile),
                      'r') as fr:
                labelList = fr.readlines()
                for label in labelList:
                    label = label.strip().split()
                    # YOLO format
                    x = float(label[1])
                    y = float(label[2])
                    w = float(label[3])
                    h = float(label[4])

                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * W
                    y1 = (y - h / 2) * H
                    x2 = (x + w / 2) * W
                    y2 = (y + h / 2) * H

                    box_count += 1
                    width = w * W
                    height = h * H
                    dataset['annotations'].append({
                        'area':
                        width * height,
                        # COCO format
                        'bbox': [x1, y1, width, height],
                        'category_id':
                        int(label[0]) + 1,
                        'id':
                        box_count,
                        'image_id':
                        img_count,
                        'iscrowd':
                        0,
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                    })
        else:
            print("path not exists: " +
                  os.path.join(originLabelsDir, dataset_name, txtFile))
    print("frame: {}, box: {}".format(img_count, box_count))

with open(json_name, 'w') as f:
    json.dump(dataset, f)