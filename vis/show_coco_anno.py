import pathlib
import sys
import cv2

import numpy as np
import todd
import torchvision
from tqdm import trange

sys.path.insert(0, '')

dataset = torchvision.datasets.CocoDetection(
    root='/train2017/',
    annFile='/annotations/instances_train2017.json',
)
out = pathlib.Path('visual_train')
out.mkdir(parents=True, exist_ok=True)
sample_ratio = 500

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from todd.base import BBoxesXYXY, BBoxTuple

Color = Tuple[int, int, int]


def draw_annotation(
    image: np.ndarray,
    bbox: BBoxTuple,
    color: Color,
    text: str,
) -> None:
    assert image.flags.contiguous
    lt = (int(bbox[0]), int(bbox[1]))
    rb = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, lt, rb, color, thickness=3)
    cv2.putText(
        image,
        text=text,
        org=lt,
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1,
        color=color,
        thickness=2,
    )


def draw_annotations(
    image: np.ndarray,
    bboxes: BBoxesXYXY,
    colors: Sequence[Color],
    texts: Sequence[str],
) -> None:
    for bbox, color, text in zip(bboxes, colors, texts):
        draw_annotation(image, bbox, color, text)



for i in trange(int(len(dataset) / sample_ratio)):
    image, target = dataset[i * sample_ratio]
    cv_image = np.array(image)[:, :, ::-1].copy()
    bboxes = todd.BBoxesXYXY(todd.BBoxesXYWH([_['bbox'] for _ in target]))
    texts = [dataset.coco.cats[_['category_id']]['name'] for _ in target]

    colors = [(255, 0, 0) for text in texts]
    draw_annotations(
        cv_image,
        bboxes,
        colors,
        texts,
    )
    assert cv2.imwrite(f'{out}/{dataset.ids[i*sample_ratio]:012d}.png',
                       cv_image)
