import time 
import os
import re
import argparse
import smtplib
from email.mine.text import MINEText
from email.header import Header


#log_path='/home/mic_dachuang/B/mmdetection/work_dir_new_4_14/faster_adagrad/20200421_015049.log'
log_path='20200409_202738.log'
model_path=''
config_path=''

current_epoch=0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',required=True,type=str)
    parser.add_argument('model_path',required=True, type=str)
    parser.add_argument('config_path',required=True, type=str)
    args = parser.parse_args()
    
    return args


def send_mail(msg):
    


def current_log():
    with open(log_path,'r') as f:
        log=f.readlines()
    return log[-1]

def parse(log):
    global current_epoch
    if re.search('workflow:',log)==None:
        if re.search('Epoch',log)==None:
            return 0
        else:
            e=int(re.search('Epoch.+?\d\d?',log).group()[7:])
            if not e==current_epoch:
                current_epoch=e
                return 3
            else: return 2
    else:
        print('Watched:',log)
        return 1
def check():
    started=0
    log=current_log()
    status=parse(log)
    if not status:
        print('Not started.')
    elif status==1:
        if not started: 
            print('Started.')
            send_mail('training started, '+log)
        started=1
    elif status==3:
        print('Epoch ',current_epoch,' Started, more info:\n',log)
        send_mail('Epoch '+str(current_epoch)+' Started,\n'+log)
        test(current_epoch)


def test(epoch):
    model=model_path+'epoch_'+str(epoch-1)+'.pth'
    result=os.popen('pyhton tools/test.py '+config_path+' '+model+' --eval bbox')
    print(result)
    wechat_push(result)

de

def timer():
    global log_path, model_path, config_path
    args=parse_args()
    log_path=args.log_path
    model_path=args.model_path
    config_path=args.config_path
    
    while(True):
        check()
        time.sleep(10)
timer()
        
    

