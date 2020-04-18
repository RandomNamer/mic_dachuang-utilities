import os.path as osp
import xml.etree.ElementTree as ET
import json
from glob import glob
from tqdm import tqdm
#from pillow import Image
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

classes = ['abnomal']
label_ids = {name: i + 1 for i, name in enumerate(classes)}
print(label_ids)

def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        category_id = 1
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w*h
        annotation.append({
                        "area": area,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": [xmin, ymin, w, h],
                        "category_id": category_id,
                        "id": anno_id,
                        "ignore": 0})
        anno_id += 1
    return annotation, anno_id

def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        
        image = cv2.imread(img_path)
        w, h,_ = image.shape
        img_name = osp.basename(img_path)
        img_info = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img_info)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        
        for ann_info in annos:
            bbox = ann_info['bbox']
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2] + bbox[0] - 1, bbox[3] + bbox[1] - 1), (0, 0, 255), 10)
            cv2.putText(image, f'{bbox[2]},{bbox[3]}',(bbox[0], bbox[1] + 1), font, 1, (0, 255, 0), 2)
        cv2.imwrite('image_box/'+img_name,image)
        img_id += 1
        
        

    categories = []
    for k,v in label_ids.items():
        categories.append({"name": k, "id": v})
        
    final_result = {"images": images,
                    "annotations": annotations, 
                    "categories": categories}
    
    with open(out_file, 'w') as f:
        json.dump(final_result, f)
    return annotations

#xml_path = '/home/mic_dachuang/coco_data/val/xml_val'
#img_path = '/home/mic_dachuang/coco_data/val/image_val'
xml_path='/Volumes/Samsung_T5/大创/data/clue_coco/val2017'
img_path='/Volumes/Samsung_T5/大创/data/clue_coco/val2017'
print('processing {} ...'.format("xml format annotations"))
cvt_annotations(img_path, xml_path, '/Volumes/Samsung_T5/大创/data/clue_coco/annotations/instances_val2017.json')
print('Done!')