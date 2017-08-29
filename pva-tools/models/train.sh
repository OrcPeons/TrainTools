#!/bin/bash
# delete data cache
cd './data'
cache='./cache'
if [ -d "$cache" ]; then
echo "Delete the cache"
rm -r "$cache"
fi
#
vocPath="VOCdevkit2007"
if [ -d "$vocPath" ]; then  
echo "Delete the symbolicLink"
rm -r "$vocPath"  
fi 
echo "Create SymbolicLink VOCdevkit2007"
# should change the directory as yours
ln -s  "/home/cvrsg/20170712/wdm/VOCBread" "$vocPath"
cd '..'
# make the label
python ./data/makeMain.py
# train
python train_demo.py 2>&1 | tee log.txt 

#python ./train_net.py
#python ./recognition/train_net.py --gpu 0 --solver model/train/solver.prototxt --cfg model/train/train.yml  --iters 10000 --imdb voc_2007_trainval --weights model/train/original_train.model
