#!/usr/bin/env python
# -*- coding:utf-8 -*-


from __future__ import print_function, division
import sys
import numpy as np
from scipy import misc
import cv2
import cv2.cv as cv
import numpy as np
from scipy import ndimage, misc
from skimage import data

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import os
import copy

XML_EXT = '.xml'

class PascalVocReader:

    def __init__(self, filepath):
        self.polygons = []
        self.polygonsName = []
        self.filepath = filepath
        self.verified = False
        self.ImageSize = ()
        self.parseXML()

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    # parse XML to get polygons
    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        self.root = xmltree
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            #polygon = object_iter.find("polygon")
            bndbox = object_iter.find('bndbox')
            name = object_iter.find('name')
            self.addPolygon(bndbox,name)
        sizeNode = xmltree.find('size')
        width = eval(sizeNode.find('width').text)
        height = eval(sizeNode.find('height').text)
        self.ImageSize = (width, height)
        return True

    # get polygons
    def getPolygons(self):
        return self.polygons
    def getImageSize(self):
        return self.ImageSize

    def getNames(self):
        return self.polygonsName
    # form xml get polygon and append it in polygons
    def addPolygon(self, polygon, name):
        #points = []
        #for point in polygon.findall('point'):
           # points.append(eval(point.text)) #eval
        xmin = eval(polygon.find('xmin').text)
        ymin = eval(polygon.find('ymin').text)
        xmax = eval(polygon.find('xmax').text)
        ymax = eval(polygon.find('ymax').text)
        points = [(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)]

        self.polygons.append(points)
        self.polygonsName.append(name.text)

    # edit the polygons(data augmentation)
    def editPolygons(self,polygons, names):
        # xml_tree = copy.deepcopy(self.root)
        self.polygons = polygons
        self.polygonsName = names
        # num = 0
        # for object_iter in xml_tree.findall('object'):
        #     # remove polygon
        #     bndbox = object_iter.find('bndbox')
        #     object_iter.remove(bndbox)
        #     polygon_iter = SubElement(object_iter, 'polygon')
        #     id = 0
        #     for point in polygons[num]:
        #         point_iter = SubElement(polygon_iter, 'point')
        #         point_iter.text = str(polygons[num][id])
        #         id = id + 1
        #     assert len(polygon)==id, 'numbers of points are not matched'
        #     num = num + 1 #the last element's id is len(polygon)-1
        #     return xml_tree
    def editImageSize(self,imageSize):
        self.ImageSize = imageSize
    # edit the filename(data augmentation)
    def editFilename(self, filename):
        fl_name = self.root.find('filename')
        fl_name.text = filename
    # save xml(data augmentation)
    def saveXML(self, xml_tree, targetFile=None):
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(xml_tree)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

    # convert polygon to bndbox(pascal voc xml)
    def convertPolygon2BndBox(self,polygon):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in polygon:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)
        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1
        if ymin < 1:
            ymin = 1
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def savePascalVocXML(self,targetFile=None):
        pascal_voc_tree = copy.deepcopy(self.root)
        num = 0
        sizeNode = pascal_voc_tree.find('size')
        widthNode = sizeNode.find('width')
        widthNode.text = str(self.ImageSize[0])
        heightNode = sizeNode.find('height')
        heightNode.text = str(self.ImageSize[1])
        for object_iter in pascal_voc_tree.findall('object'):
            pascal_voc_tree.remove(object_iter)
        num = 0
        for polygon in self.polygons:
            object_iter = SubElement(pascal_voc_tree, 'object')
            name_iter = SubElement(object_iter, 'name')
            name_iter.text = str(self.polygonsName[num])
            pose_iter = SubElement(object_iter, 'pose')
            pose_iter.text = str('Unspecified')
            truncated_iter = SubElement(object_iter, 'truncated')
            truncated_iter.text = str(0)
            difficult_iter = SubElement(object_iter, 'difficult')
            difficult_iter.text = str(0)
            bnd_box = self.convertPolygon2BndBox(polygon)
            bnd_box_iter = SubElement(object_iter, 'bndbox')
            xmin = SubElement(bnd_box_iter, 'xmin')
            xmin.text = str(bnd_box[0])
            ymin = SubElement(bnd_box_iter, 'ymin')
            ymin.text = str(bnd_box[1])
            xmax = SubElement(bnd_box_iter, 'xmax')
            xmax.text = str(bnd_box[2])
            ymax = SubElement(bnd_box_iter, 'ymax')
            ymax.text = str(bnd_box[3])
            num = num + 1

        # for object_iter in pascal_voc_tree.findall('object'):
        #     bnd_box = self.convertPolygon2BndBox(self.polygons[num])
        #     #bndbox = SubElement(object_iter, 'bndbox')
        #     bndbox = object_iter.find("bndbox")
        #     xmin = bndbox.find('xmin')
        #     xmin.text = str(bnd_box[0])
        #     ymin = bndbox.find('ymin')
        #     ymin.text = str(bnd_box[1])
        #     xmax = bndbox.find('xmax')
        #     xmax.text = str(bnd_box[2])
        #     ymax = bndbox.find('ymax')
        #     ymax.text = str(bnd_box[3])
        #     num = num + 1
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(pascal_voc_tree)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


def get_pad_length(polygon, height, width):
    polygon_width = -polygon[0][0] + polygon[2][0]
    polygon_height = -polygon[0][1] + polygon[1][1]
    # short_side = 0
    rate = 0.25
    if(polygon_height < polygon_width):
        short_side = polygon_height
    else:
        short_side = polygon_width

    # pad_length = 0
    pad_length = int(short_side * rate)
    # if(pad_length > pad_length1):
    #     pad_length = pad_length1
    pad_length2 = polygon[0][0] - 1
    if (pad_length > pad_length2):
        pad_length = pad_length2
    pad_length3 = polygon[0][1] - 1
    if (pad_length > pad_length3):
        pad_length = pad_length3
    pad_length4 = height - polygon[3][1] - 1
    if (pad_length > pad_length4):
        pad_length = pad_length4
    pad_length5 = width - polygon[3][0] - 1
    if (pad_length > pad_length5):
        pad_length = pad_length5
    #
    # if((polygon[0][0] - pad_length) < 0):
    #     pad_length = polygon[0][0] - 1
    # if((polygon[0][1] - pad_length) < 0):
    #     pad_length = polygon[0][1] - 1
    # if((polygon[3][0] + pad_length) > polygon_height):
    #     pad_length = polygon_height - polygon[3][0] - 1
    # if((polygon[3][1] + pad_length) > polygon_width):
    #     pad_length = polygon_width - polygon[3][1] - 1
    return pad_length, polygon_width, polygon_height

def get_resize_rate(polygon, pad_length):
    resize_rate = 0.66
    width = -polygon[0][0] + polygon[2][0] + pad_length * 2
    height = -polygon[0][1] + polygon[1][1] + pad_length * 2

    if(width > height):
        long_side = width
        short_side = height
        is_long_side = 0
    else:
        long_side = height
        short_side = width
        is_long_side = 1
    ratio = float(long_side) / float(short_side)

    if(ratio < 1.5):
        if (width * resize_rate > 528):
            resize_rate = float(528) / float(width)
        if (height * resize_rate > 320):
            resize_rate = float(320) / float(height)
        is_square = 1
    else:
        is_square = 0
        if(is_long_side):
            if(height * resize_rate > 640):
                resize_rate = float(640) / float(height)
            if(width * resize_rate > 1056):
                resize_rate = float(1056) / float(width)
        else:
            if(width * resize_rate > 1056):
                resize_rate = float(1056) / float(width)
            if(height * resize_rate > 320):
                resize_rate = float(320) / float(height)

    return resize_rate, is_square, is_long_side


def crop_create_img(blank_img, sample_img, xml_name, img_num):
    reader = PascalVocReader(os.path.join(xml_dir, xml_name))
    polygons = reader.getPolygons()
    new_polygons = []
    names = reader.getNames()
    new_names = []
    imageSize = reader.getImageSize()
    width = imageSize[0]
    height = imageSize[1]
    # resize_rate = 0.66
    # blank_img_copy = blank_img
    # blank_img_copy = np.zeros(blank_img.shape[0], blank_img.shape[1], blank_img.shape[2])
    # blank_img_copy = cv2.copyMakeBorder(blank_img)
    blank_img_copy = copy.deepcopy(blank_img)

    is_filled = [0, 0, 0, 0]
    num = 0
    for polygon in polygons:

        pad_length, polygon_width, polygon_height = get_pad_length(polygon, height, width)
        crop_img = sample_img[(polygon[0][1] - pad_length):(polygon[1][1] + pad_length), (polygon[0][0] - pad_length):(polygon[2][0] + pad_length), :]
        resize_rate, is_square, is_long_side = get_resize_rate(polygon, pad_length)
        size = (int((pad_length + polygon_width) * resize_rate), int((pad_length + polygon_height) * resize_rate))
        resize_img = cv2.resize(crop_img, size, interpolation=cv2.INTER_AREA)
        # b = resize_img[5, 5, 0]
        # g = resize_img[5, 5, 1]
        # r = resize_img[5, 5, 2]

        if is_square:
            new_names.append(names[num])
            num = num + 1
            new_polygon = []
            if(is_filled[0] == 0):
                x1 = int((528 - resize_img.shape[1]) * 0.5)
                x2 = x1 + resize_img.shape[1]
                y1 = int((320 - resize_img.shape[0]) * 0.5)
                y2 = y1 + resize_img.shape[0]
                b = resize_img[5, 5, 0]
                if(abs(b - 156) > 20):
                    b = 156
                g = resize_img[5, 5, 1]
                if (abs(g - 199) > 20):
                    g = 199
                r = resize_img[5, 5, 2]
                if (abs(b - 250) > 20):
                    r = 250
                blank_img_copy[0:320, 0:528] = [b, g, r]
                blank_img_copy[y1:y2, x1:x2, :] = resize_img
                point1 = (x1 + int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point1)
                point2 = (x1 + int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point2)
                point3 = (x2 - int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point3)
                point4 = (x2 - int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point4)
                is_filled[0] = 1
            elif(is_filled[1] == 0):
                x1 = int((528 - resize_img.shape[1]) * 0.5 + 528)
                x2 = x1 + resize_img.shape[1]
                y1 = int((320 - resize_img.shape[0]) * 0.5)
                y2 = y1 + resize_img.shape[0]
                b = resize_img[5, 5, 0]
                if (abs(b - 156) > 20):
                    b = 156
                g = resize_img[5, 5, 1]
                if (abs(g - 199) > 20):
                    g = 199
                r = resize_img[5, 5, 2]
                if (abs(b - 250) > 20):
                    r = 250
                blank_img_copy[0:320, 528:1055] = [b, g, r]
                blank_img_copy[y1:y2, x1:x2, :] = resize_img
                point1 = (x1 + int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point1)
                point2 = (x1 + int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point2)
                point3 = (x2 - int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point3)
                point4 = (x2 - int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point4)
                is_filled[1] = 1
            elif(is_filled[2] == 0):
                x1 = int((528 - resize_img.shape[1]) * 0.5)
                x2 = x1 + resize_img.shape[1]
                y1 = int((320 - resize_img.shape[0]) * 0.5 + 320)
                y2 = y1 + resize_img.shape[0]
                b = resize_img[5, 5, 0]
                if (abs(b - 156) > 20):
                    b = 156
                g = resize_img[5, 5, 1]
                if (abs(g - 199) > 20):
                    g = 199
                r = resize_img[5, 5, 2]
                if (abs(b - 250) > 20):
                    r = 250
                blank_img_copy[321:639, 0:528] = [b, g, r]
                blank_img_copy[y1:y2, x1:x2, :] = resize_img
                point1 = (x1 + int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point1)
                point2 = (x1 + int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point2)
                point3 = (x2 - int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point3)
                point4 = (x2 - int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point4)
                is_filled[2] = 1
            else:
                x1 = int((528 - resize_img.shape[1]) * 0.5 + 528)
                x2 = x1 + resize_img.shape[1]
                y1 = int((320 - resize_img.shape[0]) * 0.5 + 320)
                y2 = y1 + resize_img.shape[0]
                b = resize_img[5, 5, 0]
                if (abs(b - 156) > 20):
                    b = 156
                g = resize_img[5, 5, 1]
                if (abs(g - 199) > 20):
                    g = 199
                r = resize_img[5, 5, 2]
                if (abs(b - 250) > 20):
                    r = 250
                blank_img_copy[321:639, 528:1055] = [b, g, r]
                blank_img_copy[y1:y2, x1:x2, :] = resize_img
                point1 = (x1 + int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point1)
                point2 = (x1 + int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point2)
                point3 = (x2 - int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
                new_polygon.append(point3)
                point4 = (x2 - int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
                new_polygon.append(point4)
                is_filled[3] = 1
            new_polygons.append(new_polygon)
        else:
            # blank_img_copy2 = blank_img
            # blank_img_copy2 = cv2.copyMakeBorder(blank_img)
            # blank_img_copy2 = np.zeros(blank_img.shape[0], blank_img.shape[1], blank_img.shape[2])
            blank_img_copy2 = copy.deepcopy(blank_img)
            b = resize_img[5, 5, 0]
            if (abs(b - 156) > 20):
                b = 156
            g = resize_img[5, 5, 1]
            if (abs(g - 199) > 20):
                g = 199
            r = resize_img[5, 5, 2]
            if (abs(b - 250) > 20):
                r = 250
            blank_img_copy2[:, :] = [b, g, r]
            new_polygons_long = []
            new_names_long = []
            new_polygon = []
            # if is_long_side:
            x1 = int((1056 - resize_img.shape[1]) * 0.5)
            x2 = x1 + resize_img.shape[1]
            y1 = int((640 - resize_img.shape[0]) * 0.5)
            y2 = y1 + resize_img.shape[0]
            blank_img_copy2[y1:y2, x1:x2, :] = resize_img
            point1 = (x1 + int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
            new_polygon.append(point1)
            point2 = (x1 + int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
            new_polygon.append(point2)
            point3 = (x2 - int(pad_length * resize_rate), y1 + int(pad_length * resize_rate))
            new_polygon.append(point3)
            point4 = (x2 - int(pad_length * resize_rate), y2 - int(pad_length * resize_rate))
            new_polygon.append(point4)
            new_polygons_long.append(new_polygon)
            new_names_long.append(names[num])
            num = num + 1
            reader.editImageSize((1056, 640))
            reader.editPolygons(new_polygons_long, new_names_long)
            #reader.savePascalVocXML('/home/kyxu/Python_study/bread-crop/data/pascal2/' + str(img_num) + '.xml')
            #cv2.imwrite('/home/kyxu/Python_study/bread-crop/data/processed/' + str(img_num) + '.jpg', blank_img_copy2)
            reader.savePascalVocXML(xml_save + str(img_num) + '.xml')
            cv2.imwrite(img_save + str(img_num) + '.jpg', blank_img_copy2)
            img_num = img_num + 1

        if(is_filled[3] == 1):
            is_filled = [0, 0, 0, 0]
            reader.editImageSize((1056, 640))
            reader.editPolygons(new_polygons, new_names)
            #reader.savePascalVocXML('/home/kyxu/Python_study/bread-crop/data/pascal2/' + str(img_num) + '.xml')
            #cv2.imwrite('/home/kyxu/Python_study/bread-crop/data/processed/' + str(img_num) + '.jpg', blank_img_copy)
            reader.savePascalVocXML(xml_save + str(img_num) + '.xml')
            cv2.imwrite(img_save + str(img_num) + '.jpg', blank_img_copy)
            img_num = img_num + 1
            # blank_img_copy = blank_img
            # blank_img_copy = cv2.copyMakeBorder(blank_img)
            # blank_img_copy = np.zeros(blank_img.shape[0], blank_img.shape[1], blank_img.shape[2])
            blank_img_copy = copy.deepcopy(blank_img)
            new_polygons = []
            new_names = []

    if len(new_polygons):
        reader.editImageSize((1056, 640))
        reader.editPolygons(new_polygons, new_names)
        #reader.savePascalVocXML('/home/kyxu/Python_study/bread-crop/data/pascal2/' + str(img_num) + '.xml')
        #cv2.imwrite('/home/kyxu/Python_study/bread-crop/data/processed/' + str(img_num) + '.jpg', blank_img_copy)
        reader.savePascalVocXML(xml_save + str(img_num) + '.xml')
        cv2.imwrite(img_save + str(img_num) + '.jpg', blank_img_copy)
        img_num = img_num + 1
    return img_num

if __name__ == '__main__':
    # path to read
    voc_dir = "/home/kyxu/Python_study/bread-crop/data"
    xml_dir = os.path.join(voc_dir, 'pascal')
    img_dir = os.path.join(voc_dir, 'img')
    imgs = os.listdir(img_dir)
    blank_img = cv2.imread('blank_img.jpg')
    # path to save
    xml_save = 'pascal/'
    img_save = 'img/'
    img_num = 0

    for img_name in imgs:
        xml_name = img_name[0: -4] + '.xml'
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_num = crop_create_img(blank_img, img, xml_name, img_num)


