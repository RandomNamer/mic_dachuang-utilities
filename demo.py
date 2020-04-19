from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import cv2
import re
import json
from glob import glob  
from tqdm import tqdm

#os.environ['CUDA_VISIBLE_DEVICES'] = "3"

config_file = '/home/mic_dachuang/B/mmdetection/modified_configs/ssd512_coco.py'
checkpoint_file = '/home/mic_dachuang/B/mmdetection/work_dir_new_4_14/ssd512_default/epoch_5.pth'
#config_file='/Volumes/Samsung_T5/大创/data/coco_new/test2017/'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


picture='/home/mic_dachuang/B/test/test2017'
annotation='/home/mic_dachuang/B/test/test2017/'
output='/home/mic_dachuang/B/test/out/'

for img in tqdm(glob(picture + '/*.jpg')):
  fn=re.search(r'[-\w]*.jpg',img).group()
#with open(picture) as f:
  # test a single image and show the results
  #img = mmcv.imread('test.jpg')#, which will only load it once
  result = inference_detector(model, img)
  # visualize the results in a new window
  show_result(img, result, model.CLASSES,0.3,0,False,output+fn)
  i=cv2.imread(output+fn)
#Read existing annotations and draw them out
  json_path=annotation+'/'+fn[:-4]+'.json'
  with open(json_path) as j:
    all=json.load(j)
  for ann in all:
    pt1,pt2=(int(ann['x']),int(ann['y'])),(int(ann['x']+ann['w']),int(ann['y']+ann['h']))
   # print(pt1,pt2,img)
    cv2.rectangle(i,pt1,pt2,(0,0,255),1)
    font=cv2.FONT_HERSHEY_PLAIN
    cv2.putText(i,'Reference',pt1,font,1,(0,0,255),1)
    cv2.imwrite(output+fn,i)
print('Done.')


