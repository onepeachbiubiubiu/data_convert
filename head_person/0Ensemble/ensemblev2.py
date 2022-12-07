import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
def getdata(pth):
    with open(pth,'r') as f:
        data = f.readlines()
        return data

def listBoxes(pathDir,image_path,fich):
    boxes = []  # list that will contain all the squares of each xml
    prob = 0.5# if there is no probability, this default value is assigned
    print(fich)
    (nameFich, extension) = os.path.splitext(fich)

    if (extension == ".txt"):#we stay with those who are xmls and we go through them looking for a box
        boxes=[]
        equalFiles = glob.glob(pathDir+'/*/'+fich)#
        for f in equalFiles:
            j = 0  
            doc = getdata(f)
            img_path = os.path.join(image_path,fich.replace("txt","jpg"))
            img = cv2.imread(img_path)
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
                if prob < 0.5:
                    continue 
                boxes.append([class_, xmin, ymin, xmax, ymax, prob])
                j = j+1
        return nameFich,boxes,len(equalFiles)
    return None

def getPrim(boxes):
    # import pdb;pdb.set_trace()
    scores = np.array(boxes)[:,-1]
    return np.argsort(-scores)

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def uneBoundingBoxes(boxesAll):
    # boxesAll = list(set((boxesAll)))
    boundingBox=[]
    listBox = []
    l=len(boxesAll)
    while(l>0):
        #[排序]
        # import pdb;pdb.set_trace()
        boxesAll = list(np.array(boxesAll)[getPrim(boxesAll)])
        boxPrim= boxesAll[0]
        listBox.append(boxPrim)
        boxesAllXmls1=boxesAll[1:]
        # import pdb;pdb.set_trace()
        removearray(boxesAll,boxPrim)
        for box in boxesAllXmls1:
            if boxPrim[0]==box[0] and bb_intersection_over_union(boxPrim[1:5], box[1:5]) > 0.5:
                listBox.append(box)
                removearray(boxesAll,box)

        boundingBox.append(listBox)
        listBox = []
        l=len(boxesAll)
    return boundingBox



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

def nonMaximumSuppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [0]
    # initialize the list of picked indexes
    pick = []
    probFinal = 0
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1].astype(float)
    y1 = boxes[:, 2].astype(float)
    x2 = boxes[:, 3].astype(float)
    y2 = boxes[:, 4].astype(float)
    prob = boxes[:, 5].astype(float)
    for l in prob:
        probFinal = probFinal+l
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(prob)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick], probFinal