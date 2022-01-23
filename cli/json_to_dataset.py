# -*- coding: UTF-8 -*-
# !H:\Anaconda3\envs\new_labelme\python.exe
import argparse
import json
import math
import os
import os.path as osp
import warnings 
from PIL import Image, ImageDraw

import cv2
import numpy as np
 

def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def labelme_shapes_to_label(w,h, shapes, line_width=10):

    cls_list = []
    ins_list = []
    for shape in shapes:
        cls = np.zeros((w,h), dtype=np.int32)
        ins = np.zeros((w,h), dtype=np.int32)
        points = shape['points']
        shape_type = shape['shape_type']
        mask = shape_to_mask((w,h), points, shape_type, line_width = line_width)
        print(shape.keys())
        cls_id = shape["cls_id"]
        ins_id = shape["ins_id"]

        cls[mask] = cls_id
        ins[mask] = ins_id
        cls_list.append(cls)
        ins_list.append(ins)

    return cls_list, ins_list

# from sys import argv

def save_image(img, filename, type, root=".", color_map=(11093 ,7229 ,20887 )):
    savefile = os.path.join(root,filename+"_"+type+".png")
    savefile2 = os.path.join(root,filename+"_"+type+"_color.png")

    image_color = np.zeros((img.shape[0],img.shape[1],3), dtype=np.int32)
    image_color[:,:,0] = img*color_map[0] % 256
    image_color[:,:,1] = img*color_map[1] % 256
    image_color[:,:,2] = img*color_map[2] % 256

    cv2.imwrite(savefile, img)
    cv2.imwrite(savefile2, image_color)
    pass

def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
 
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('-c','--class_name', default=None)
    parser.add_argument('-n','--number',default=None)
    parser.add_argument('-m','--mask',default=None)
    parser.add_argument('-w','--line_width',type=int,default=10)
    parser.add_argument('-u','--updata',action='store_true', default=False)
    args = parser.parse_args()
 
    json_file = args.json_file
    lineszie = int(args.line_width)
    nmask = args.mask

    if not os.path.isdir(json_file):
        print( 'no such file!!')
        return

    list_path = os.listdir(json_file)
    print(list_path[0:100])

    for file in list_path:
        
        (fname, extension) = os.path.splitext(file)
        # picpath = os.path.join(json_file,fname+'.jpg')
        
        
        if (nmask is not None) and (not fname.startswith(nmask)):
            continue

        if (extension == '.json'):
            # try:
            print(file)
            path = os.path.join(json_file, file)  
            data = json.load(open(path))
            h, w = data['imageWidth'], data['imageHeight']
    
            cls_list, ins_list = labelme_shapes_to_label(w, h, data['shapes'], lineszie)

            cls_img = np.max( np.concatenate([clss[:,:,None] for clss in cls_list], axis=2), axis=2)
            ins_img = np.max( np.concatenate([inss[:,:,None] for inss in ins_list], axis=2), axis=2)

            if not osp.exists(json_file + '\\' + 'class_mask_png'):
                os.mkdir(json_file + '\\' + 'class_mask_png')
            saveroot = json_file + '\\' + 'class_mask_png'

            save_image(cls_img,fname,'cls',saveroot)
            save_image(ins_img,fname,'ins',saveroot)
                
    
 
 
if __name__ == '__main__':
    # base64path = argv[1]
    main()
