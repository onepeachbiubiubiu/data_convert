# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tqdm import tqdm
import mmcv
import numpy as np

parser = argparse.ArgumentParser(description='Test yolo data.')
parser.add_argument('-d',
                    dest='dataroot',
                    help='The data root of coco dataset.',
                    required=True)
parser.add_argument(
    '-o',
    dest='outdir',
    help='The output directory of coco semi-supervised annotations.',
    required=True)
parser.add_argument('-f',
                    dest='fold',
                    help='The percentage of labeled data in the training set.',
                    required=True)

args = parser.parse_args()
data_root = args.dataroot
output = args.outdir
fold = args.fold


def split_coco(data_root, output, fold):
    """Split COCO data for Semi-supervised object detection.

    Args:
        data_root (str): The data root of coco dataset.
        out_dir (str): The output directory of coco semi-supervised
            annotations.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
    """
    def save_anns(name, images, anns):
        sub_anns = dict()
        sub_anns['images'] = images
        sub_anns['annotations'] = anns['annotations']
        sub_anns['type'] = anns['type']
        sub_anns['categories'] = anns['categories']
        # sub_anns['info'] = anns['info']

        mmcv.mkdir_or_exist(output)
        print(f'{output}/{name}.json')
        mmcv.dump(sub_anns, f'{output}/{name}.json')

    # import pdb;pdb.set_trace()
    # set random seed with the fold
    ann_file = osp.join(data_root, 'annotations/instances_train2018.json')
    anns = mmcv.load(ann_file)
    # import pdb;pdb.set_trace()
    image_list = anns['images']
    aver_num = int(len(image_list) / fold) + 1
    tempnum = 0

    for i in tqdm(range(fold)):
        if tempnum + aver_num < len(image_list):
            tempimage = image_list[tempnum:tempnum + aver_num]
            tempnum = tempnum + aver_num
        else:
            tempimage = image_list[tempnum:len(image_list)]
            tempnum = len(image_list)
        # save labeled and unlabeled
        labeled_name = f'instances_train2017.{i}'
        save_anns(labeled_name, tempimage, anns)


if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    split_coco(data_root, output, int(fold))
