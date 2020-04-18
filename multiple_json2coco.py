import os.path as osp
import os
import json
from glob import glob  
from tqdm import tqdm
from PIL import Image
import re
import shutil
#import cv2
#font = cv2.FONT_HERSHEY_SIMPLEX
#classes = ['clue_cells']
#label_ids = {name: i + 1 for i, name in enumerate(classes)}
#FastWrite:
cat={"categories": [{"name": "clue_cells", "id": 1}]}

def parse_json(json_path,img_path,drop_path):
    with open(json_path,'r')as f:
        ann=json.load(f)
    if ann==[]:
        empty=True
    else: empty=False
    return [ann,empty]

def merge(ann_path,img_path,output_path,drop_path=None):
    if not drop_path: drop_path=img_path+'/dropped/'
    if not osp.exists(drop_path): os.mkdir(drop_path)
    img_id = 1
    anno_id = 1
    images=[]
    annotations=[]
    categories=cat
    drop_cnt=0
    for img_path in tqdm(glob(img_path + '/*.jpg')):
         fn=re.search(r'[-\w]*.jpg',img_path).group()
         json_path=ann_path+'/'+fn[:-4]+'.json'
         
         res=parse_json(json_path,img_path,drop_path)
         if not res[1]:
             img=Image.open(img_path)
             [h,w]=img.size
             images.append({"file_name": fn, "height": h, "width": w, "id": img_id})
             AllAnnotaions=parse_json(json_path,img_path,drop_path)[0]
             for ann in AllAnnotaions:
                 area=ann['w']*ann['h']
                 bbox=[ann['x'],ann['y'],ann['w'],ann['h']]
                 annotations.append({"area":area , "iscrowd": 0, "image_id": img_id, "bbox": bbox, "category_id": 1, "id": anno_id, "ignore": 0})
                 anno_id=anno_id+1
             img_id=img_id+1 
         else: 
             shutil.copy(json_path,drop_path)
             shutil.copy(img_path,drop_path)
             os.remove(json_path)
             os.remove(img_path)
             print("Image "+img_path+" has no associated annotation, and it was moved to ."+drop_path)
             drop_cnt=drop_cnt+1
    print("\nProcessed",img_id-1,'images and',anno_id-1,'annotations,Dropped ',drop_cnt," images.")
    with open(output_path,'w') as f:
        json.dump({"images":images,"annotations":annotations,"categories":categories},f)

merge("/Volumes/Samsung_T5/大创/data/coco_new/val2017","/Volumes/Samsung_T5/大创/data/coco_new/val2017","/Volumes/Samsung_T5/大创/data/coco_new/instances_val2017.json","/Volumes/Samsung_T5/大创/data/coco_new/dropped/")