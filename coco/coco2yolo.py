import json
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test yolo data.')
parser.add_argument('-j', help='JSON file', dest='json', required=True)
parser.add_argument('-o',
                    help='path to output folder',
                    dest='out',
                    required=True)

args = parser.parse_args()

json_file = args.json
output = args.out


class COCO2YOLO:
    def __init__(self):
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in tqdm(self.labels['images']):
            id = image['id']
            file_name = image['file_name']
            # import pdb;pdb.set_trace()
            if file_name.find('\\') > -1:
                print(file_name)
                file_name = file_name[file_name.index('\\') + 1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno(self, images_info):
        anno_dict = dict()
        anno_pth = ".../mmdetection/dev_res"
        # merge some json
        for jsons in os.listdir(anno_pth):
            json_pth = os.path.join(anno_pth, jsons)
            f = open(json_pth, 'r')
            data = json.load(f)
            for anno in tqdm(data):
                bbox = anno['bbox']
                # import pdb;pdb.set_trace()
                image_id = anno['image_id']
                category_id = anno['category_id']

                image_info = images_info.get(image_id)
                image_name = image_info[0]
                img_w = image_info[1]
                score = anno['score']
                img_h = image_info[2]
                yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

                anno_info = (image_name, category_id, yolo_box, score)
                anno_infos = anno_dict.get(image_id)
                if not anno_infos:
                    anno_dict[image_id] = [anno_info]
                else:
                    anno_infos.append(anno_info)
                    anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(
            map(lambda x: x['name'],
                sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        length = len(anno_dict)
        now_len = 0
        p = tqdm(total=length)
        # import pdb;pdb.set_trace()
        # p.start()
        for k, v in anno_dict.items():
            p.update(now_len)
            file_name = v[0][0].split(".")[0] + ".txt"
            # import pdb;pdb.set_trace()
            file_path = file_name.split("/")
            os.makedirs(os.path.dirname(
                os.path.join(output, file_path[-2], file_path[-1])),
                        exist_ok=True)
            with open(os.path.join(output, file_path[-2], file_path[-1]),
                      'w',
                      encoding='utf-8') as f:
                # print(k, v)
                # import pdb;pdb.set_trace()
                for obj in v:
                    category_id = 0
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box.append('{:.6f}'.format(obj[3]))
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')
            now_len += 1
        p.close()


if __name__ == '__main__':
    c2y = COCO2YOLO()
    c2y.coco2yolo()