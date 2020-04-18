from PIL import Image
import os
#img_path="/Volumes/Samsung_T5/大创/util/cropped_1.jpg"
#img_path="/Volumes/Macintosh HD/Users/zzy/Downloads/clue/JISNJGLFY19011004200040014.jpg"
root_dir="/home/mic_dachuang/data/B/clue"
dest_dir="/home/mic_dachuang/data/B/sliced/"
img_cnt=0
slc_cnt=0
for file in os.listdir(root_dir):
    img_path = root_dir + '/' + file
    if file[0]=='.': continue
    file_name=file[0:len(file)-4]
    img=Image.open(img_path)
    print(file,' ',img.size)
    crop_factor=[(0,0,1024,1024),(1024,0,2048,1024),\
        (0,1024,1024,2048),(1024,1024,2048,2048),\
        (2048,0,3072,1024),(2048,1024,3072,2048)]
    if not img.size==(1024,1024):
        if img.size==(3072,2048):
            cnt=0
            slc_cnt=slc_cnt+6
            for crp in crop_factor:
                cropped=img.crop(crp)
                cropped.save(dest_dir+file_name+'_'+str(cnt)+".jpg")
                cnt=cnt+1
        elif img.size==(2048,2048):
            slc_cnt=slc_cnt+4
            for i in range(0,3):
                cropped=img.crop(crop_factor[i])
                cropped.save(dest_dir+file_name+'_'+str(i)+".jpg")
        elif: print("      Outlier Resolution:",img.size)
    else: slc_cnt=slc_cnt+1
    img_cnt=img_cnt+1
print("Processed "+str(img_cnt)+" imgaes in total, generated "+str(slc_cnt)" images.")

    
