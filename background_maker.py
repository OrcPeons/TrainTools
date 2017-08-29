# -*- coding: utf-8 -*-
import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2
import random

# read xml find boundingbox,then random-crop to generate negtive samples whose iou are low

def getiou(box1,box2):#box = [x1, y1, x2, y2]
    iou = -1
    if (box1[0]-box2[2])*(box1[2]-box2[0])<0 and (box1[1]-box2[3])*(box1[3]-box2[1])<0:
        if (box1[0]-box2[0])*(box1[2]-box2[2]) > 0:
            w = min(abs(box1[0] - box2[2]), abs(box1[2] - box2[0]))
        else:
            w = min(abs(box1[0] - box1[2]), abs(box2[2] - box2[0]))
        if (box1[1] - box1[3]) * (box2[1] - box2[3]) > 0:
            h = min(abs(box1[1] - box2[3]), abs(box1[3] - box2[1]))
        else:
            h = min(abs(box1[1] - box1[3]), abs(box2[3] - box2[1]))
        area1 = float((abs(box1[2]-box1[0])) * float(abs(box1[3]-box1[1])))
        area2 = float((abs(box2[2]-box2[0])) * float(abs(box2[3]-box2[1])))
        s = min(area1,area2)
        iou = float(w)* float(h) / s
    else:
        iou = 0
    return iou

def back_box(img,boxes,backnum): #box = [x1, y1, x2, y2]
    total = 0
    back_boxes = []
    while(total<backnum):
        # random rect
        w = 225 + 25 * random.randint(0, 8)  # background size:
        h = 150 + 25 * random.randint(0, 8)
        x = random.randint(1, 1279 - w)
        y = random.randint(1, 719 - h)
        rect = [x, y, x+w, y+h]
        # iou flag
        flag = 0
        for box in boxes:
            iou = getiou(rect,box)
            if iou >= 0.25:
                flag=1
                break

        if flag==0:
            back_boxes.append(rect)
            total += 1

    return back_boxes



# read xml
xmlpath = '/home/wurui/Desktop/bread_label/VOC2007/Annotations'
filelist = os.listdir(xmlpath)

# img size = 1280*720
# background size : [150*150 - 525*525]
back_num = 3 # background generated per img
period = 10 # stepsize of img to generate background
count = 0
max = 900

ind = 0
for file in filelist:
    if ind >= max:
        break

    if count%period == 0:
        DOMTree = xml.dom.minidom.parse(xmlpath + '/' + file)
        collection = DOMTree.documentElement
        # element path
        fname = collection.getElementsByTagName("filename")[0]
        filename = fname.childNodes[0].data
        imgpath = '/home/wurui/Desktop/bread_label/VOC2007/JPEGImages/' + str(filename) + '.jpg'
        img = cv2.imread(imgpath)
        # objects
        objs = collection.getElementsByTagName("object")
        boxes = []
        for obj in objs:
            xmin = obj.getElementsByTagName("xmin")[0]
            x1 = int(xmin.childNodes[0].data)
            ymin = obj.getElementsByTagName("ymin")[0]
            y1 = int(ymin.childNodes[0].data)
            xmax = obj.getElementsByTagName("xmax")[0]
            x2 = int(xmax.childNodes[0].data)
            ymax = obj.getElementsByTagName("ymax")[0]
            y2 = int(ymax.childNodes[0].data)
            box = [x1, y1, x2, y2]
            boxes.append(box)

        backgrounds = back_box(img, boxes, back_num)
        for rect in backgrounds:
            copyimg = img[rect[1]:rect[3], rect[0]:rect[2]]  # copyimg = img[y1:y2, x1:x2]
            savename = '/home/wurui/Desktop/bread_label/backs/' + 'b' + str(ind) + '.jpg'
            cv2.imwrite(savename, copyimg)
            ind += 1
            print ind


    count+=1



