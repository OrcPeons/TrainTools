#!/usr/bin/env python
import os
import numpy as np

this_dir = os.path.dirname(__file__)

xmlfilepath = os.path.join(this_dir,'VOCdevkit2007/VOC2007/Annotations');
txtsavepath = os.path.join(this_dir,'VOCdevkit2007/VOC2007/ImageSets/Main/');

trainval_percent = 0.9;
train_percent = 0.8;

xmlfile = os.listdir(xmlfilepath);
numOfxml = len(xmlfile);
# trainval vs test
numOfxml_perm = np.random.permutation(numOfxml);
trainval_ind = numOfxml_perm[:int(np.floor(numOfxml*trainval_percent))];
trainval = np.sort(trainval_ind);
test = np.sort(np.setdiff1d(numOfxml_perm,trainval));
# train vs val
trainvalsize = len(trainval);
trainvalsize_perm = np.random.permutation(trainvalsize);
train_ind = trainvalsize_perm[:int(np.floor(trainvalsize*train_percent))];
train = np.sort(trainval[train_ind]);
val = np.sort(np.setdiff1d(trainval,train));
# write to file
ftrainval= open(os.path.join(txtsavepath,'trainval.txt'),'w');
ftest= open(os.path.join(txtsavepath,'test.txt'),'w');
ftrain= open(os.path.join(txtsavepath,'train.txt'),'w');
fval= open(os.path.join(txtsavepath,'val.txt'),'w');

for i in range(numOfxml):
    if i in trainval:
        ftrainval.write(xmlfile[i][:-4]+'\n')
        if i in train:
            ftrain.write(xmlfile[i][:-4]+'\n')
        else:
            fval.write(xmlfile[i][:-4]+'\n')
    else:
        ftest.write(xmlfile[i][:-4]+'\n')

ftrainval.close();
ftest.close();
ftrain.close();
fval.close();
