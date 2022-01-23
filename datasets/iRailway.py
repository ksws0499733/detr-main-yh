# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import sys
print(sys.path)
sys.path.append(r"E:\G\论文——铁路轨道识别\代码\detr-main")
# from ..panopticapi import utils
from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms
import os




class iRailway:
    def __init__(self, img_folder, ann_folder=None, label_folder=None, transforms=None, return_masks=True):

        # .jpg == image
        # .json == image/info
        # cls.png, ins.png == image/info/class_mask_png

        if ann_folder is None:
            ann_folder = os.path.join(img_folder,"Info")
        
        if label_folder is None:
            label_folder = os.path.join(ann_folder,"class_mask_png")

        img_list = os.listdir(img_folder)
        ann_list = os.listdir(ann_folder)
        label_list = os.listdir(label_folder)

        self.img_list = []
        self.ann_list = []
        self.cls_list = []
        self.ins_list = []

        for annfile in ann_list:
            if annfile.endswith('.json'):
                # name,ext = os.path.splitext(annfile)
                imgfile = annfile.replace('.json', '.jpg')
                clsfile = annfile.replace('.json', '_cls.png')
                insfile = annfile.replace('.json', '_ins.png')
                if imgfile in img_list and clsfile in label_list and insfile in label_list:
                    self.img_list.append(os.path.join(img_folder,imgfile))
                    self.cls_list.append(os.path.join(label_folder,clsfile))
                    self.ins_list.append(os.path.join(label_folder,insfile))

                    with open(os.path.join(ann_folder,annfile), 'r') as f:
                        ann_info = json.load(f)
                        self.ann_list.append(ann_info)

        print('img_list', len(self.img_list))

        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):

        img_path = self.img_list[idx]        
        ins_path = self.ins_list[idx]
        # cls_path = self.cls_list[idx]

        ann_info = self.ann_list[idx]


        img = Image.open(img_path).convert('RGB')
        w, h = img.size


        masks = np.asarray(Image.open(ins_path), dtype=np.uint32)
        # print(np.unique(masks))

        masks = rgb2id(masks)#颜色 = ID
        # print(np.unique(masks))
        

        ids = np.array([0] + [ann['ins_id'] for ann in ann_info['shapes']])
        
        # print(ids)
        masks = masks == ids[:, None, None]

        masks = torch.as_tensor(masks, dtype=torch.uint8)



        labels = torch.tensor([0]+[ann['cls_id'] for ann in ann_info['shapes']], dtype=torch.int64)

        target = {}
        # target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
       
        target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_list)

    def get_height_and_width(self, idx):
        ann_info = self.ann_list[idx]
        height = ann_info['imageHeight']
        width = ann_info['imageWidth']
        return height, width


def build(image_set, args):
    dataset_root = Path(args.iRailway_path)

    # PATHS = {
    #     "train": os.path.join(dataset_root,"train"),
    #     "val": os.path.join(dataset_root,"val"),
    # }
    PATHS = {
        "train": dataset_root,
        "val": dataset_root
    }

    img_folder = PATHS[image_set]
    dataset = iRailway(img_folder,
                           transforms=make_coco_transforms(image_set), 
                           return_masks=args.masks)

    return dataset


if __name__ == '__main__':
    image_set = "train"
    img_folder = r"E:\dataSet\all_dataset\other01"
    dataset = iRailway(img_folder,
                           transforms=make_coco_transforms(image_set)
                           )    

    from torch.utils.data import DataLoader, DistributedSampler
    import util.misc as utils
    from models.my_marcher_globelpanic import targer_cat, mask_iou_cost_dice
    data_loader_train = DataLoader(dataset, batch_size=3,collate_fn=utils.collate_fn)
    for samples, targets in data_loader_train:
        src, mask = samples.decompose()
        print("samples src",src.shape)
        print("samples mask",mask.shape)
        for i in range(len(targets)):
            print("targets",targets[i].keys())
            print("masks",targets[i]['masks'].shape)
            print("labels",targets[i]['labels'].shape)
            print("boxes",targets[i]['boxes'])
        tgt_mask = targer_cat(targets)
        tgt_mask = tgt_mask.to(src)
        print("targer_cat",tgt_mask.shape)

        b,c,h,w = src.shape

        # tgt_mask = torch.nn.functional.interpolate(tgt_mask[:, None], size=(h//4,w//4),
        #         mode="bilinear", align_corners=False)
        # print("tgt_mask",tgt_mask.shape)
        # # tgt_mask.squeeze()
        # tgt_mask = tgt_mask[:, 0]
        # print("tgt_mask_flat",tgt_mask.shape)

        mask_cost = mask_iou_cost_dice(mask,tgt_mask)
        print("mask_cost",mask_cost.shape)


        print("")
    
