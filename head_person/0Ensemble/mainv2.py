import ensemblev2
import argparse
import numpy as np
import glob
import os
import math
import cv2
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to the labels of images")
ap.add_argument("-o", "--option", required=True,help="option to the ensemble: affirmative, consensus or unanimous")
ap.add_argument("-i", "--image", required=True,help="option to the ensemble: affirmative, consensus or unanimous")
args = vars(ap.parse_args())
image_path = args["image"]
#read the arguments
datasetPath= args["dataset"]
option = args["option"]
# import ipdb;ipdb.set_trace()
#we get a list that contains as many pairs as there are xmls in the first folder, these pairs indicate first the
#name of the xml file and then contains a list with all the objects of the xmls
print(datasetPath) 
dirs = os.listdir(datasetPath)
total_txt = os.listdir(os.path.join(datasetPath,dirs[0]))
#n张图的box集合
for nombree in tqdm(total_txt):
    nombre,lis,numFich = ensemblev2.listBoxes(datasetPath,image_path,nombree)
    assert nombre+".txt" == nombree
    pick = []
    resul = []
    #we check if the output folder exists
    output_path = os.path.join(datasetPath,"..","output")
    os.makedirs(output_path,exist_ok=True)
    file = open(os.path.join(output_path,nombre+".txt"), "w")
    filename = nombre  # we look for the root of our xml
    print(nombre+" "+str(len(lis)))
    box = ensemblev2.uneBoundingBoxes(lis)
    # box应该是一张图:格式[[某个group里面一个方法给出的一个框,,],[],[],...]
    #apply the corresponging technique
    print(nombre+" "+str(len(box)))
    img_path = os.path.join(image_path,nombre+".jpg")
    img = cv2.imread(img_path)
    w = img.shape[1]
    h = img.shape[0]
    for rectangles in box:
        # rectangles是一个group
        list1 = []
        for rc in rectangles:
            list1.append(rc)
        pick = []
        if option == 'consensus':
            if len(np.array(list1))>=math.ceil(numFich/2):
                pick,prob = ensemblev2.nonMaximumSuppression(np.array(list1), 0.3)
                # pick[0][5] = prob/numFich

        elif option == 'unanimous':
            if len(np.array(list1))==numFich:
                pick,prob = ensemblev2.nonMaximumSuppression(np.array(list1), 0.3)
                # pick[0][5] = prob / numFich

        elif option == 'affirmative':
            pick,prob = ensemblev2.nonMaximumSuppression(np.array(list1), 0.3)
            # pick[0][5] = prob / numFich

        if len(pick)!=0:
            print(len(pick))
            resul.extend(pick)
            
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
