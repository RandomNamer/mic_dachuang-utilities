import os
from glob import glob

work_dir='/home/mic_dachuang/B/mmdetection/work_dir_new_4_14/ssd512_default'
config='/home/mic_dachuang/B/mmdetection/modified_configs/ssd512_coco.py'

f=open(work_dir+'testall.txt','w+')
content=''
for epoch in glob(work_dir+'/*.pth'):
    print("Testing "+epoch+'...')
    raw=os.popen('pyhton tools/test.py '+config+' '+epoch+' --eval bbox').readlines()
    result=raw[-12:0]
    print(result)
    content=content+result+'\n'
f.write(content)
f.flush()
