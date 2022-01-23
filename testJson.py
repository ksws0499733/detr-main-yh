import json
import os

filname = '000695.json'
outname = '000695_out.json'

root = '.'
label_dict = {'ballast':1,'sleeper':2,'track':3,'track_left':4,'track_right':5,'track_inner':6}


divergence_dict ={
    'ballast':('ballast','Ballast'),
    'sleeper':('sleeper','railwayline','railwayLine',"Railway","railway","RailwayLine"),
    'track':('track','track'),
    'track_left':('track_left','track_left'),
    'track_right':('track_right','track_right'),
    'track_inner':('track_inner','track_inner')
}

def lable_diver_correction(divLabel):
    for key in divergence_dict.keys():
        if divLabel in divergence_dict[key]:
            return key
    return divLabel



with open(filname) as f:
    load_dict  = json.load(f)
    if "imageData" in load_dict.keys():
        del load_dict['imageData']; # 删除键是'Name'的条目


    assert "shapes","label" in load_dict.keys()
    for idx, shape in enumerate( load_dict["shapes"]):
        if shape['label'] not in label_dict.keys():
            shape['label'] = lable_diver_correction(shape['label'])

        updict = {"ins_id":idx,
                "cls_id":label_dict[shape['label']]
        }
        shape.update(updict)
        print(idx,shape)

    print(load_dict)
    
    with open(outname,"w") as p:
        json.dump(load_dict,p,indent=1)