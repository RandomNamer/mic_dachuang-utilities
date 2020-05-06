from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
from glob import glob  
from tqdm import tqdm
import re
import json
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

config_file='/home/mic_dachuang/B/mmdetection/modified_configs/faster_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file='/home/mic_dachuang/B/mmdetection/work_dir_new_4_14/faster_adagrad/epoch_8.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
clue='/home/mic_dachuang/B/mmdetection/data_new_4_14/test2017/'
plain='/home/mic_dachuang/B/test/plain'
output='/home/mic_dachuang/B/test/clue.json'

result=[]


def get_bbox_num(bboxes,fn,threshold):
  global result
  if len(bboxes)>0:
      res_bbox=[]
      for i,bbox in enumerate(bboxes):
          if float(bbox[4])>threshold:
              res_bbox.append([round(float(x),2) for x in bbox])              
      #result.append('filename':fn,'bboxes':res_bbox)
    return len(res_bbox)       


def get_bbox_count(imgs):
    bbox_count={'0':[],'0.1':[],'0.2':[],'0.3':[],'0.4':[],'0.5':[],'0.6':[],'0.7':[],'0.8':[],'0.9':[]}
    for j,img in enumerate(imgs):
        fn=re.search(r'[-\w]*.jpg',img).group()
        res=inference_detector(model,img)
        bboxes=np.vstack(res)
        for threshold in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            bbox_sum=get_bbox_num(bboxes,fn,threshold)
            print('Tested image ',fn,',',j,'of ',len(imgs),bbox_sum)
            bbox_count[str(threshold).append(bbox_sum)
    return bbox_count

clue_bbox_count=get_bbox_count(glob(clue+'*.jpg')) 
plain_bbox_count=get_bbox_count(glob(plain+'*.jpg'))
sns.set_style('darkgrid')
plt.figure(figsize=(10,10))
for key in clue_bbox_count.keys():
    y=clue_bbox_count(key)
    x=range(len(y))
    plt.plot(x,y,label=key)
plt.legend()
plt.savefig('./clue_bbox_count,jpg')
print('Saved clue_bbox_count plots at different thresholds.')

plt.figure(figsize=(10,10))
for key in plain_bbox_count.keys():
    y=plain_bbox_count(key)
    x=range(len(y))
    plt.plot(x,y,label=key)
plt.legend()
plt.savefig('./plain_bbox_count,jpg')
print("Saved plain_bbox_count plots at different thresholds.")

'''
with open (output,'w+') as f:
    json.dump(result,f)
print("Done.") 
'''
