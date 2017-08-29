#!/usr/bin/env python
import os.path as osp
import os
import recognition   
import cv2
import time
import copy
import Image
import ImageDraw
import ImageFont
from utils.timer import Timer

this_dir = osp.dirname(__file__)
cfg_file = osp.join(this_dir, 'model', 'comp','bread.yml')
net_pt = osp.join(this_dir,'model', 'gh','bread.prototxt')
net_weight = osp.join(this_dir, 'output','faster_rcnn_pvanet','voc_2007_trainval','wdm_iter_130000.caffemodel')
cls_path = osp.join(this_dir, 'model','gh','name.txt')
label_path = osp.join(this_dir, 'model','gh','label.txt')
f = open(cls_path)
class_name = ['__background__']
while 1:
    name = f.readline()
    name = name.strip()
    class_name.append(name)
    if not name:
        break

class_name.pop()
CLASSES = class_name

f2 = open(label_path)
label_name = ['__background__']
while 1:
    name = f2.readline()
    name = name.strip()
    label_name.append(name)
    if not name:
        break

label_name.pop()
LABEL = label_name

def getIOU(Reframe,GTframe):
    ''' Rect = [x1, y1, x2, y2] '''
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2]-Reframe[0];
    height1 = Reframe[3]-Reframe[1];

    x2 = GTframe[0];
    y2 = GTframe[1];
    width2 = GTframe[2]-GTframe[0];
    height2 = GTframe[3]-GTframe[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height; 
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1.0/(Area1+Area2-Area);
    # return IOU
    return ratio


def delete_box_iou(old_detections,thresh=0.4):
    new_detections = copy.copy(old_detections)
    index = []
    #ioulist = []
    for i in range(len(new_detections)): # 0 -- len-1
        for j in range(i+1,len(new_detections)):
            iou = getIOU(new_detections[i][1:5],new_detections[j][1:5])
            if iou >= thresh :
                #ioulist.append(iou)
                if new_detections[i][5] >= new_detections[j][5]:  
                    index.append(j)
                else:
                    index.append(i)
    output = []
    for idx,detec in enumerate(new_detections):
        flag = 0
        for i in index:
            if idx == i:
                flag=1
        if flag == 0:
            output.append(detec)
        
    for idx in index:
        new_detections[idx]

    return output



if __name__ == '__main__':
    
    # set mode
    recognition.set_mode('gpu',0)
    #recognition.set_mode('cpu')
    # if ClASSES changed, update the test.prototxt
    recognition.change_test_prototxt(net_pt, len(CLASSES))
    #load model, just load once, or else it will be very slow
    net = recognition.load_net(cfg_file, net_pt, net_weight)
    #trigger the camera
    #image = recognition.take_picture(0)

    # just like the predefined data structure
    results = {}

    # just use a picture to test the program
    test_dir = osp.join(this_dir, 'data', 'demo','test-gh')
    #test_dir = osp.join(this_dir, 'data', 'demo','threestep','car')
    #test_dir = osp.join(this_dir, 'data', 'demo', 'threestep', 'plate')
    #test_dir = osp.join(this_dir, 'data', 'demo', 'threestep', 'platenumber')
    result_dir = osp.join(this_dir, 'data', 'demo','result')
    test_imgs = os.listdir(test_dir)
    for img_path in test_imgs:
        for cls_ind, cls in enumerate(CLASSES[1:]):
            results[cls] = 0
        image = cv2.imread(osp.join(test_dir,img_path))
        image_pil = Image.open(osp.join(test_dir,img_path))
        drawobj = ImageDraw.Draw(image_pil)

        #recognition : can loop
        timer = Timer()
        timer.tic()
        detections = recognition.detect(net, image, CLASSES,0.4)
        timer.toc()
        print ('Detection took {:.3f}s').format(timer.total_time)
    
        new_detections = delete_box_iou(detections,0.4) # iou thresh
        print 'old_detections==============',detections
        print 'new_detections==============',new_detections

        
        #output by the predefined data structure 
        for detection in new_detections:
            # get the count of breads
            results[detection[0]] = results[detection[0]] + 1
            drawobj.rectangle((int(detection[1]),int(detection[2]),int(detection[3]),int(detection[4])),outline="red")
            drawobj.rectangle((int(detection[1]),int(detection[2]),int(detection[1])+150,int(detection[2])+25),fill=128)
            ind = CLASSES.index(detection[0])
            txt = unicode(LABEL[ind],'utf-8')
            font = ImageFont.truetype('/usr/share/fonts/truetype/YaHei.Consolas.1.11b_0.ttf',18)
            drawobj.text((int(detection[1]),int(detection[2])),txt,font=font)
        #print or save the results
        print results
        rst_path = img_path
        image_pil.save(osp.join(result_dir,rst_path))
        #image_pil.show()
        #cv2.imwrite(osp.join(result_dir,rst_path), image)

