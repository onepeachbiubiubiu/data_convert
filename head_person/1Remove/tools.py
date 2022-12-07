import os
import glob
import numpy as np
import cv2
def getdata(pth):
    with open(pth,'r') as f:
        data = f.readlines()
        return data
def listBoxes(path,image_path):
    boxes = []
    doc = getdata(path)
    img = cv2.imread(image_path)
    w = img.shape[1]
    h = img.shape[0]
    for line in doc:
        line_data =line.split(" ")
        class_ = eval(line_data[0])
        xmin = (float(line_data[1])-float(line_data[3])/2)*w
        ymin = (float(line_data[2])-float(line_data[4])/2)*h
        xmax = (float(line_data[1])+float(line_data[3])/2)*w
        ymax = (float(line_data[2])+float(line_data[4])/2)*h
        prob = eval(line_data[5])
        boxes.append([class_, xmin, ymin, xmax, ymax, prob])
    return boxes,w,h

def listBoxesbyWH(path,w,h):
    boxes = []
    doc = getdata(path)
    for line in doc:
        line_data =line.split(" ")
        class_ = eval(line_data[0])
        xmin = (float(line_data[1])-float(line_data[3])/2)*w
        ymin = (float(line_data[2])-float(line_data[4])/2)*h
        xmax = (float(line_data[1])+float(line_data[3])/2)*w
        ymax = (float(line_data[2])+float(line_data[4])/2)*h
        if len(line_data) >5:
            prob = eval(line_data[5])
        else:
            prob = 1.0
        boxes.append([class_, xmin, ymin, xmax, ymax, prob])
    return boxes

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def IOH(head,person):
    xA = max(head[0], person[0])
    yA = max(head[1], person[1])
    xB = min(head[2], person[2])
    yB = min(head[3], person[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    headArea = (head[2] - head[0] + 1) * (head[3] - head[1] + 1)
    ioh = interArea / float(headArea)
    return ioh
def IOHS(heads,persons):
    iohs = []
    for head in heads:
        head_ioh = []
        for person in persons:
            ioh = IOH(head,person)
            head_ioh.append(ioh)
        iohs.append(head_ioh)
    return iohs

def writefile(file_path,resul,w,h):
    file = open(file_path,'w')
    for res in resul:
        file.write(str(res[0]))
        file.write(" ")
        file.write(str((res[1]+res[3])/2/w))
        file.write(" ")
        file.write(str((res[2]+res[4])/2/h))
        file.write(" ")
        file.write(str((res[3]-res[1])/w))
        file.write(" ")
        file.write(str((res[4]-res[2])/h))
        file.write(" ")
        file.write(str(res[5]))
        file.write("\n")
    file.close()
 