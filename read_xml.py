# -*- coding: utf-8 -*-
import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2

# img save path depend on labels
savepath = '/home/wurui/Desktop/bread_label/train/'
label = ['DNZ','NNBD','YRPT','QDNLHP','NJKS','NXSYMB']
labelfile = []
for clas in label:
    savedir = savepath+clas
    if os.path.exists(savedir)==False:
        os.makedirs(savedir)

# read xml
xmlpath = '/home/wurui/Desktop/bread_label/VOC2007/Annotations'
filelist = os.listdir(xmlpath)
count = 0
for file in filelist:
    DOMTree = xml.dom.minidom.parse(xmlpath+'/'+file)
    collection = DOMTree.documentElement
    # element path
    fname = collection.getElementsByTagName("filename")[0]
    filename = fname.childNodes[0].data
    imgpath = '/home/wurui/Desktop/bread_label/VOC2007/JPEGImages/'+str(filename)+'.jpg'
    img = cv2.imread(imgpath)
    # objects
    objs = collection.getElementsByTagName("object")
    for obj in objs:
        xmin = obj.getElementsByTagName("xmin")[0]
        x1 = int(xmin.childNodes[0].data)
        ymin = obj.getElementsByTagName("ymin")[0]
        y1 = int(ymin.childNodes[0].data)
        xmax = obj.getElementsByTagName("xmax")[0]
        x2 = int(xmax.childNodes[0].data)
        ymax = obj.getElementsByTagName("ymax")[0]
        y2 = int(ymax.childNodes[0].data)
        copyimg = img[y1:y2,x1:x2]
        # save img
        lab = obj.getElementsByTagName("name")[0]
        cls = lab.childNodes[0].data  # label class name
        for name in label:
            if name==str(cls):
                saveimg = savepath + name + '/' + str(count) + '.jpg'
                cv2.imwrite(saveimg,copyimg)
                count+=1







