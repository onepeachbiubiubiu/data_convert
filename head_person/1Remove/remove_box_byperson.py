import argparse
import numpy as np
import glob
import os
import math
import cv2
import tools
from tqdm import tqdm
ap = argparse.ArgumentParser()
ap.add_argument("-dh", "--datasetofhead", required=True,help="path to the labels of head")
ap.add_argument("-dp", "--datasetofperson", required=True,help="path to the labels of person")
ap.add_argument("-i", "--image", required=True,help="path to the image")
ap.add_argument("-o", "--output", required=True,help="output dir")
ap.add_argument("-s", "--score_thresh", required=True,help="score_thresh")#0.5

args = vars(ap.parse_args())
imagePath = args["image"]
#read the arguments
headPath= args["datasetofhead"]
personPath= args["datasetofperson"]
output = args["output"]
os.makedirs(output,exist_ok=True)
thresh = eval(args["score_thresh"])
heads = os.listdir(headPath)

for txt in tqdm(heads):
    head_path = os.path.join(headPath,txt)
    person_path = os.path.join(personPath,txt)
    img_path = os.path.join(imagePath,txt.replace("txt","jpg"))
    out_path = os.path.join(output,txt)
    headdata,w,h = tools.listBoxes(head_path,img_path)
    persondata = tools.listBoxesbyWH(person_path,w,h)
    headdata,persondata = np.array(headdata),np.array(persondata)
    iohs = tools.IOHS(headdata[:,1:5],persondata[:,1:5])
    iohs = np.array(iohs)
    ioh = np.argmax(iohs,axis =0)
    iohh = np.max(iohs,axis =0)>0.4
    ioh = ioh[iohh]
    ioh = list(set(ioh))
    # may have many person to one head
    if len(ioh) < persondata.shape[0]-1:
        ioscore = np.max(iohs,axis =1)
        ioid = np.argmax(iohs,axis =1)
        head_score = np.array(headdata)[:,-1]*ioscore*np.array(persondata)[ioid][:,-1]
        head_valid = list(np.where(head_score>thresh)[0])
        ioh.extend(head_valid)
        head_valid = list(np.where(np.array(headdata)[:,-1]>0.8)[0])
        ioh.extend(head_valid)
        ioh = list(set(ioh))
    final_head = []
    for i,head in enumerate(headdata):
        if i in ioh:
            # temphead = list(head[:-1])
            # temphead.append(head_score[i])
            final_head.append(head)
    tools.writefile(out_path,final_head,w,h)

