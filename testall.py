import os
from glob import glob

work_dir='/home/mic_dachuang/mmdetection/B/work_dir/faster_rcnn_48epoch'
config='/home/mic_dachuang/mmdetection/B/faster_rcnn_r50_fpn_1x.py'

f=open(work_dir+'/testall.txt','w+')
content=''
for epoch in glob(work_dir+'/*.pth'):
    print('Testing ',epoch,'...')
    raw=os.popen('python tools/test.py '+config+' '+epoch+' --eval bbox').readlines()
    result=raw[-12:]
    print(result)
    content=content+epoch+':\n'+str(result)+'\n'
f.write(content)
f.flush()
f.close()
