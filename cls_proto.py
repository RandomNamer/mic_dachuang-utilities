plain_config='/home/mic_dachuang/B/test/plain.json'
clue_config='/home/mic_dachuang/B/test/clue.json'

additional_info='confidence threshold=0.6'

import json
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure(figsize=(12,8))

from sklearn import svm,naive_bayes,neighbors,tree
configs={
    'svm': svm.SVC(),\
    'decision_tree':tree.DecisionTreeClassifier(),
    'naive_gaussian': naive_bayes.GaussianNB(), \
    'naive_mn':naive_bayes.MultinomialNB(),\
    'K_neighbor' : neighbors.KNeighborsClassifier(),
}

def load_data():
    plain=[]
    clue=[]
    with open(plain_config,'r') as fp:
        plain=json.load(fp)
    with open(clue_config,'r') as fc:
        clue=json.load(fc)
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
        x.append([num_bboxes,avg_size,avg_conf])
        bn.append(num_bboxes)
        cf.append(avg_conf)
        sz.append(avg_size)
    print(bn,'\n',cf)
    plt.scatter(sz,cf,marker='o',c='green',alpha=0.6,label='None Clue Cells')
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
        x.append([num_bboxes,avg_size,avg_conf])
        print(num_bboxes,avg_conf)
        bn.append(num_bboxes)
        cf.append(avg_conf)
        sz.append(avg_size)
    print(bn,'\n',cf)
    plt.scatter(sz,cf,marker='o',c='red',alpha=0.5,label='Clue Cells')
    plt.legend()
    plt.savefig('./plt.png')
    return x,y


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

def train_test(cfg):
    cfg.fit(x_train,y_train.ravel())
    score=cfg.score(x_test,y_test.ravel())
    print("Score: ",score)

for cfg_key in configs.keys():
    print("Method: ",cfg_key)
    train_test(configs[cfg_key])



