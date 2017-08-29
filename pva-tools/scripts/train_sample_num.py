# -*- coding: utf-8 -*-
import os
from xml.dom.minidom import parse
import xml.dom.minidom

keys = ['yexiangmianbao','zhishishunisanwenzhi','jingdouhongdoubao','hongdoushaomianbao',
'xiangsuanmianbao','putaochuanmianbao','quanmaihetaomianbao','fushishan','rulaonaiyoupai',
'moxigerousongjuan','bilishihetaoquan','shenfudeguaizhang','weiyenanaiyoubang','pisaduo',
'meishiregoujuan','zaoanshanghai','nailaosanjiao','niurousong','haidaochuan','andelupingzhuangheimeiguojiang',
'alabang_he','qiaokeliquqibinggan_he','lanmeiquqi_huangyouquqi','masaikebinggan','yeziqiu_he',
'andelupingzhuangcaomeiguojiang','laopobing_nuomi','pushidanta','laopobing_dousha','moxigenaixiangmianbao',
'huangjinrulaomianbao','biantaorenhetaocuibing','laopobing_zishu','yikousu','lianghuochaibangmianbao',
'manyuemeirulaomianbao','pamasenzhishibang','fengtangniujiaomianbao','maomaochongmianbao','qiaokelidangao',
'yuanweifagun','ganlanfagun','suanrongfagun','jiaotangqiancengsu','hetaobao','tiziduomianbao','rousonghuotui',
'niujiaomianbao','zhishiregou','xinruanshousimianbao','rulaobaodao','quanmaimanyuemeimianbao','boluomianbao',
'mochanailu','niunaitusimianbao','jinzhuanmianbao','tangchuntusimianbao','dannaitusimianbao','quanmaitusimianbao',
'daboluotusimianbao','doushatusimianbao','naisutusimianbao','yerongtusimianbao','hongzaodangao','biantaorendangao',
'tianshirulaodangao','bojuesanwenzhi','bingxueqiyuan','naiyougege','bingxuexiari','suannaibafei','boshidundangao',
'heisenlin','jingdiantianrannaiyoudangao','mochadangao','tilamisumusi','mangguomusi','lanmeimusi','ailisitianpinbei',
'zhishijiaxinqiaokelidangao','zhishijiaxindangao','nailaonapolun','huorunmeiguiyingtaoguolifengweisuanniunai','huoruncaomeisangrenguolifengweisuanniunai',
'huorunhuangtaomangguoguolifengweisuanniunai','yishinailaodangao','chaorouzhishidangao','xuemeiniang',
'xianzhachengzhi','banshurulaoliangli','guoleshilanmeicaomeiheijialunkexiguoni','guoleshicaomeixiangjiaokexiguoni','jidanxianshusanwenzhi']
values = []
for i in range(94):
    values.append(0)

dic = dict(zip(keys, values))
 
xml_path = '/home/cvrsg/pvanet/pva-faster-rcnn-old/data/VOCdevkit2007/VOC2007/Annotations'
xmls = os.listdir(xml_path)
for xml_file in xmls:
    DOMTree = xml.dom.minidom.parse(xml_path+'/'+xml_file)
    collection = DOMTree.documentElement
    # objects
    objs = collection.getElementsByTagName("object")
    for obj in objs:
        lab = obj.getElementsByTagName("name")[0]
        cls = lab.childNodes[0].data  # label class name
        dic[cls]+=1

print dic
xml_path2 = '/home/cvrsg/pvanet/pva-faster-rcnn-old/data/VOCdevkit2007/VOC2007/img_crop_backup_20170716/Annotations'
xmls2 = os.listdir(xml_path2)
for xml_file2 in xmls2:
    DOMTree2 = xml.dom.minidom.parse(xml_path2+'/'+xml_file2)
    collection2 = DOMTree2.documentElement
    # objects
    objs2 = collection2.getElementsByTagName("object")
    for obj2 in objs2:
        lab2 = obj2.getElementsByTagName("name")[0]
        cls2 = lab2.childNodes[0].data  # label class name
        print cls2
        dic[cls2]+=1

print dic
f=open('sample_num.txt','w')
for k in keys:
    f.write(k+' = '+str(dic[k])+'\n')
    
    
