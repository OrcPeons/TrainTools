#!/usr/bin/env python
from recognition import aux_tools
import os.path as osp
import os
import shutil

this_dir = osp.dirname(__file__)

CLASSES = ['__background__','xiangsuanmianbao','putaochuanmianbao','quanmaihetaomianbao','bilishihetaoquan','laopobing_nuomi','laopobing_dousha','laopobing_zishu','hetaobao','rousonghuotui']
#'DNZ','NNBD','YRPT','QDNLHP','NJKS','NXSYMB')

net_pt = osp.join(this_dir, 'model', 'recognition','bread.prototxt')
net_pt_svd = osp.join(this_dir, 'model', 'comp','bread_merge_svd.prototxt')
net_weight = osp.join(this_dir, 'output','faster_rcnn_pvanet','voc_2007_trainval','wdm_iter_90000.caffemodel')
#net_weight = osp.join(this_dir, 'model', 'test_compress','bread_iter_100000.caffemodel')
cfg_file = osp.join(this_dir, 'model', 'recognition', 'bread.yml')

if __name__== '__main__':
    aux_tools.change_test_prototxt(net_pt,len(CLASSES))
    aux_tools.change_test_prototxt(net_pt_svd, len(CLASSES))
    output_pt,output_weight = aux_tools.gen_merged_model(net_pt, net_weight)
    aux_tools.compress_net(output_pt,net_pt_svd,output_weight)
    #remove template files
    os.remove(output_pt)
    os.remove(output_weight)
    #copy cfg file
    # output directory
    out_dir = os.path.dirname(net_pt_svd)
    #configure file basename
    cfg_name = os.path.basename(cfg_file)
    #to check whether a .yml file in output directoty
    file_list = os.listdir(out_dir)
    exist_flag = False
    for file in file_list:
        if file.endswith('.yml'):
            flag = True
    #copy .yml
    if osp.exists(cfg_file) and exist_flag == False:
        out_cfg = osp.join(out_dir, cfg_name)
        shutil.copy(cfg_file,out_cfg)
