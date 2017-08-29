#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os.path as osp
from recognition import train_net

this_dir = osp.dirname(__file__)

solver_prototxt = osp.join(this_dir, 'model', 'train','solver.prototxt')
#pretrained_model = osp.join(this_dir, 'model', 'train','original_train.model')
pretrained_model = osp.join(this_dir,'model','train','original_train.model')
cfg_file = osp.join(this_dir, 'model', 'train','train.yml')
train_prototxt = osp.join(this_dir, 'model', 'train','train.prototxt')
cls_path = osp.join(this_dir, 'model','data','VOCdevkit2007','VOC2007','name.txt')

f = open(cls_path)
class_name = []
while 1:
    name = f.readline()
    name = name.strip()
    class_name.append(name)
    if not name:
        break

class_name.pop()
CLASSES = tuple(class_name)


if __name__ == '__main__':
    # numbers of iteration
    #max_iters = 220000

    #train_net.set_mode('cpu')
    train_net.set_mode('gpu',0)
    # train the model
    train_net.train(classes = CLASSES,
                    cfg_file = cfg_file, solver_proto = solver_prototxt, train_prototxt = train_prototxt,
                    pretrained_model = pretrained_model, max_iters = 200000, imdb_name = 'voc_2007_trainval', set_cfgs=None)

    #train_net.train(CLASSES,
    #      cfg_file, solver_prototxt, train_prototxt, pretrained_model, max_iters , 'voc_2007_trainval')
