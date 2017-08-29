#!/usr/bin/env python
import os.path as osp
import os
import recognition   
import cv2
import time
import copy
from utils.timer import Timer

this_dir = osp.dirname(__file__)
cfg_file = osp.join(this_dir, 'model', 'wdm','bread.yml')
#net_pt = osp.join(this_dir, 'model', 'recognition','bread.prototxt')
net_pt = osp.join(this_dir, 'model', 'wdm','test-wdm.prototxt')
net_weight = osp.join(this_dir, 'output','faster_rcnn_pvanet','voc_2007_trainval','wdm_iter_90000.caffemodel')
cls_path = osp.join(this_dir, 'data','VOCdevkit2007','VOC2007','name.txt')
save_path = osp.join(this_dir, 'data','demo','result')

f = open(cls_path)
class_name = ['__background__']
while 1:
    name = f.readline()
    name = name.strip()
    class_name.append(name)
    if not name:
        break

class_name.pop()
CLASSES = tuple(class_name)




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
    test_dir = osp.join(this_dir, 'data', 'demo','test')
    result_dir = osp.join(this_dir, 'data', 'demo','result')
    test_imgs = os.listdir(test_dir)
    for img_path in test_imgs:
        for cls_ind, cls in enumerate(CLASSES[1:]):
            results[cls] = 0
        image = cv2.imread(osp.join(test_dir,img_path))
        #recognition : can loop
        timer = Timer()
        timer.tic()
        detections = recognition.detect(net, image, CLASSES,0.4)
        timer.toc()
        print ('Detection took {:.3f}s').format(timer.total_time)
    
        # wrz debug iou test
        new_detections = delete_box_iou(detections,0.4) # iou thresh
        #print 'old_detections==============',detections
        print 'new_detections==============',new_detections
        
        #output by the predefined data structure 
        for detection in new_detections:
            # get the count of breads
            results[detection[0]] = results[detection[0]] + 1
            cv2.rectangle(image,(int(detection[1]),int(detection[2])),(int(detection[3]),int(detection[4])),(255,255,255),2)
            cv2.putText(image,detection[0]+' '+str(detection[5]),(int(detection[1]),int(detection[2])),0,0.5,(0,0,255),2)
        #print or save the results
        print results
        rst_path = img_path
        #rst_path = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.jpg' #rename by time
        cv2.imwrite(osp.join(result_dir,rst_path), image)

