# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from .detr import build
from .detr import build as dbuild
from .my_detr_globelpanic import build as mybuild
# import my_detr_globelpanic

def build_model(args):
    if args.source == 'my':
        return mybuild(args)
    else:
        return dbuild(args)
