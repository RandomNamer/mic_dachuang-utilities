plain_config='/home/mic_dachuang/B/test/plain.json'
clue_config='/home/mic_dachuang/B/test/clue.json'
work_dir='/home/mic_dachuang/B/test/'
additional_info='confidence threshold=0.6'

import json
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import shutil

plt.figure(figsize=(8,6))

from sklearn import svm,naive_bayes,neighbors,tree
configs={
    'svm': svm.SVC(),\
    'decision_tree':tree.DecisionTreeClassifier(),
    'naive_gaussian': naive_bayes.GaussianNB(), \
    'naive_mn':naive_bayes.MultinomialNB(),\
    'K_neighbor' : neighbors.KNeighborsClassifier(),
}
plain=[]
clue=[]

with open(plain_config,'r') as fp:
        plain=json.load(fp)
with open(clue_config,'r') as fc:
        clue=json.load(fc)
plain_index=[]
clue_index=[]

def load_data():
    x=[]
    y=[0]*len(plain)
    bn=[]
    cf=[]
    sz=[]
    for i in plain:
        bbox_size=[]
        confidence=[]
        num_bboxes=len(i['bboxes'])
        if num_bboxes>0:
            for bbox in i['bboxes']:
                bbox_size.append((int(bbox[2])-int(bbox[0]))*(int(bbox[3])-int(bbox[1])))
                confidence.append(bbox[4])
            avg_size=np.mean(bbox_size)
            avg_conf=np.mean(confidence)
        else:
            avg_size=0
            avg_conf=0
#这里，x应该代替bboxes，重新生成一个带filename的索引类型的dict：
        plain_index.append({'filename':i['filename'],'bboxes':[num_bboxes,avg_size,avg_conf]})
        x.append([num_bboxes,avg_size,avg_conf])
        bn.append(num_bboxes)
        cf.append(avg_conf)
        sz.append(avg_size)
    print(bn,'\n',cf)
    #plt.scatter(sz,cf,marker='o',c='green',alpha=0.5,label='None Clue Cells')
    y=y+[1]*len(clue)
    bn=[]
    cf=[]
    sz=[]
    for j in clue:
        bbox_size=[]
        confidence=[]
        num_bboxes=len(j['bboxes'])
        if num_bboxes>0:
            for bbox in j['bboxes']:
                bbox_size.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
                confidence.append(bbox[4])
            avg_size=np.mean(bbox_size)
            avg_conf=np.mean(confidence)
        else:
            avg_size=0
            avg_conf=0
        clue_index.append({'filename':i['filename'],'bboxes':[num_bboxes,avg_size,avg_conf]})
        x.append([num_bboxes,avg_size,avg_conf])
        print(num_bboxes,avg_conf)
        bn.append(num_bboxes)
        cf.append(avg_conf)
        sz.append(avg_size)
        
    print(bn,'\n',cf)
    #plt.scatter(sz,cf,marker='o',c='red',alpha=0.4,label='Clue Cells')
   
    return x,y

def search(vector,y_value):
    print(y_value)
    if y_value==1: target=clue_index
    elif y_value==0: target=plain_index
    for i in target:
        #print(vector,target_x)
        if (i['bboxes']==vector).all(): return i['filename']
    return 0 

def shuffule_uni(a,b):
    assert len(a)==len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
x_raw,y_raw=load_data()
x_array=np.array(x_raw);y_array=np.array(y_raw)
x,y=shuffule_uni(x_array,y_array)
x_train,x_test=x[:1000],x[1000:]
y_train,y_test=y[:1000].reshape(-1,1),y[1000:].reshape(-1,1)

#def plot_decision_region(x,y,clf)

    

def train_test(cfg):
    cfg.fit(x_train,y_train.ravel())
    score=cfg.score(x_test,y_test.ravel())
    print("Score: ",score)

def dump_img(filename,cat):
    print('filename:',filename)
    fn_path=work_dir+'FN';fp_path=work_dir+'FP'
    if not os.path.exists(fn_path): os.mkdir(fn_path)
    if not os.path.exists(fp_path): os.mkdir(fp_path)
    if cat: shutil.copy(filename,fn_path)
    else: shutil.copy(filename,fp_path)
    return 1

def label_count(array):
    c_count=0
    p_count=0
    for i in array:
        if i==1:c_count=c_count+1
        elif i==0: p_count=p_count+1
        else: print('Not 1 or 0')
    return (c_count,p_count)

for cfg_key in configs.keys():
    print("Method: ",cfg_key)
    train_test(configs[cfg_key])
print('Now set svm classification result as output...')
configs['svm'].fit(x_train,y_train.ravel())
y_predict=configs['svm'].predict(x_test)
assert len(y_predict)==len(y_test)
print(label_count(y_test))
count=0
FN_count=0
FP_count=0
for p in range(len(y_predict)):
    if y_predict[p]!=y_test[p]: 
        if y_predict[p]==1: FP_count=FP_count+1
        elif y_predict[p]==0: FN_count=FN_count+1
        fn=search(x_test[p],y_test[p])
        if fn!=0: dump_img(fn,y_test[p])
        else: print('Not found.')
        count=count+1
print(len(y_test),'images tested, ',count,'deviant images.')
print("False Negative: ",FN_count,'False Positive: ',FP_count)





