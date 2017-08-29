import os
import shutil

f = open('test.txt')
imgs = os.listdir('JPEGImages')
while 1:
    line = f.readline()
    line = line.strip('\n')
    for img in imgs:
        name = os.path.splitext(img)[0]
        if line == name:
            oldname = os.path.join('JPEGImages',img)
            newname = os.path.join('test',img)
            print newname
            shutil.copyfile(oldname,newname)
    if not line:
        break

