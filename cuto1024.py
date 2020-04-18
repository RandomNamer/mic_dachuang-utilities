####接口

####locimg 是要切的图片路径
####locgt 是gt路径
####aimnimpath 是切好的图片发现是阴性视野图放哪的路径
####loc_IMG是切好的图片发现是阳性视野图放哪的路径
####loc_GT是切好的图片发现是阳性视野图他的json放哪的路径   ##我的的json,你们用xml的话自己转一下
####xmlpath 你们注意下 这个是视野图对应的xml的路径，自己改一下
####156行的比例0.4 你们根据自己的图像调，我用0.4卡的阈值
###注意73行

import numpy as np
import time
import os
import cv2
from glob import glob
import xml.etree.ElementTree as ET
import json
import shutil


def xml2json(xmlpath ,jsonpath):
    flag = 0
    res = []
    pcelllist = ['abnormal','hsil','HSIL','ASC-H','asc-h']
    tree = ET.parse(xmlpath)
    objs = tree.findall('object')
    if objs:
        flag = 1
        for obj in objs:
            bbx = {}
            bbox = obj.find('bndbox')
            name = obj.find('name').text
            if name in pcelllist:
                x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text) > 0 else 0
                y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text) > 0 else 0
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                w = x2-x1
                h = y2-y1
                bbx['x'] = x1
                bbx['y'] = y1
                bbx['w'] = w
                bbx['h'] = h
                bbx['class'] = name
                if x1 >= x2 or y1 >= y2:
                    continue
                else:
                    res.append(bbx)
            # boxes = np.append(boxes, np.expand_dims([x1, y1, x2, y2], axis=0).astype(np.uint16), axis=0)
    if flag:
        with open(jsonpath,'w')as f:
            json.dump(res,f)
    return flag
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

locimg = '.'+os.sep+'clue_images'+os.sep
locgt = '.'+os.sep+'clue_xmls'+os.sep
loc_IMG = '.'+os.sep+'pi'+os.sep#
loc_GT = '.'+os.sep+'pg'+os.sep
aimnimpath = '.'+os.sep+'ni'+os.sep



mkdir(loc_GT)
mkdir(loc_IMG)
mkdir(aimnimpath)

imgpaths = glob(os.path.join(locimg, '*.jpg'))
for imgpath in imgpaths:
    ## Windows:
    #imgname = imgpath.split('')[-1].split('.')[0]
    ## Linux or Unix:
    imgname = imgpath.split('/')[-1].split('.')[0]
    xmlpath = locgt+imgname+'.xml'
    aimpimpath = loc_IMG+imgname+'.jpg'
    aimgtpath = loc_GT+imgname+'.json'
    img= cv2.imread(imgpath)
    (h,w,c) = img.shape
    hnum = int(h/1024)
    wnum = int(w/1024)
    if hnum == 1 and wnum == 1:

        flag = xml2json(xmlpath,aimgtpath)
        print(flag)
        if flag == 1:
            shutil.copy(imgpath, aimpimpath)
        else:
            shutil.copy(imgpath, aimnimpath)

    else:
        print(imgname+'*'*20)
        # pcelllist = ['abnormal', 'hsil', 'HSIL', 'ASC-H', 'asc-h']
        tree = ET.parse(xmlpath)
        objs = tree.findall('object')
        res = []
        if objs:
            for obj in objs:
                bbx = {}
                bbox = obj.find('bndbox')
                name = obj.find('name').text
                # if name in pcelllist:
                x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text) > 0 else 0
                y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text) > 0 else 0
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                w = x2 - x1
                h = y2 - y1
                bbx['x'] = x1
                bbx['y'] = y1
                bbx['w'] = w
                bbx['h'] = h
                bbx['class'] = name
                if x1 >= x2 or y1 >= y2:
                    continue
                else:
                    res.append(bbx)

        for i in range(wnum):
            for j in range(hnum):
                x1 = 1024*i
                y1 = 1024*j
                x2 = x1+1024
                y2 = y1+1024
                saveimg = img[y1:y2,x1:x2,:]
                # print(saveimg.shape)
                savename = imgname+'-'+str(i*hnum+j+1)
                if res == []:
                    cv2.imwrite(aimnimpath+savename+'.jpg',saveimg)
                else:
                    jr = []

                    for bbx in res:
                        newbbx = {}
                        bx1 = bbx['x']
                        by1 = bbx['y']
                        bw = bbx['w']
                        bh = bbx['h']
                        bx2 = bx1+bw
                        by2 = by1+bh
                        lu = bx1>=x1 and bx1<=x2 and by1>=y1 and by1<=y2
                        ru = bx2>=x1 and bx2<=x2 and by1>=y1 and by1<=y2
                        lb = bx1>=x1 and bx1<=x2 and by2>=y1 and by2<=y2
                        rb = bx2>=x1 and bx2<=x2 and by2>=y1 and by2<=y2
                        if lu or ru or lb or rb:
                            fx= max(bx1*(bx1>=x1),x1)
                            fy = max(by1*(by1>=y1),y1)
                            fw = min(x2-fx,bx2-fx)
                            fh = min(y2-fy,by2-fy)
                            newbbx['x'] = fx-x1
                            newbbx['y'] = fy-y1
                            newbbx['w'] = fw
                            newbbx['h'] = fh
                            newbbx['class'] = bbx['class']
                            if (newbbx['w']*newbbx['h'])/(bw*bh)>=0.4:
                                jr.append(newbbx)

                    if jr!=[]:
                        cv2.imwrite(loc_IMG+savename+'.jpg',saveimg)
                        with open(loc_GT+savename+'.json','w')as f:
                            json.dump(jr,f)
                    else:
                        cv2.imwrite(aimnimpath+savename+'.jpg',saveimg)








