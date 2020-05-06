from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
from glob import glob  
from tqdm import tqdm
import re
import json
import numpy as np 

config_file='/home/mic_dachuang/B/mmdetection/modified_configs/faster_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file='/home/mic_dachuang/B/mmdetection/work_dir_new_4_14/faster_adagrad/epoch_8.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
clue='/home/mic_dachuang/B/mmdetection/data_new_4_14/test2017/'
plain='/home/mic_dachuang/B/test/plain'
threshold=0.6



def get_output(imgs):
    result=[]
    for j,img in enumerate(imgs):
        fn=re.search(r'[-\w]*.jpg',img).group()
        res=inference_detector(model,img)
        bboxes=np.vstack(res)
        print("Tested image ",j,'of',len(imgs),fn,len(bboxes))
        if len(bboxes)>0:
          res_bbox=[]
          for i,bbox in enumerate(bboxes):
              if float(bbox[4])>threshold:
                  res_bbox.append([round(float(x),2) for x in bbox])              
          result.append('filename':fn,'bboxes':res_bbox)
    return result
with open('./clue.json','r') as cf:
    json.dump(get_output(glob(clue+'*.jpg')),cf)
with open('./plain.json','r') as pf:
    json.dump(get_output(glob(plain+'*.jpg')),pf)
print("Done.")               
