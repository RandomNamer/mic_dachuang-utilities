import time 
import os
import re
import argparse
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import smtplib


#log_path='/home/mic_dachuang/B/mmdetection/work_dir_new_4_14/faster_adagrad/20200421_015049.log'
log_path='20200409_202738.log'
model_path=''
config_path=''
model_name=input('自定义模型名称，则为邮件发送时的送信人名称:')
mail_addr=input('想接收的邮件地址,若不输入则为默认：')

current_epoch=0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path')
    parser.add_argument('model_path')
    parser.add_argument('config_path')
    args = parser.parse_args()
    
    return args

def send_mail(log):
    global model_name, mail_addr
    msg=MIMEText(str(log),'plain','utf-8')
    usrn='smtpservicetest@163.com'
    pswd='BBSKRKQDYQFJIXKG'
    if not mail_addr: sendto='random_name@sjtu.edu.cn'
    else: send_to=mail_addr
    msg['From']='Training Brief'
    msg['To']=sendto
    msg['Subject']='Status update of model: '+model_name
    smtp=smtplib.SMTP()
    #smtp.set_debuglevel(1)
    smtp.connect('smtp.163.com',25)
    smtp.login(usrn,pswd)
    smtp.sendmail(usrn,[sendto],msg.as_string())


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
            send_mail('training started, \n'+log)
        started=1
    elif status==3:
        print('Epoch ',current_epoch,' Started, more info:\n',log)
        send_mail('Epoch '+str(current_epoch)+' Started,\n'+log)
        test(current_epoch)


def test(epoch):
    print('Testing epoch ',str(epoch-1))
    model=model_path+'/epoch_'+str(epoch-1)+'.pth'
    raw=os.popen('python tools/test.py '+config_path+' '+model_path+'/epoch_'+str(epoch-1)+'.pth --eval bbox').readlines()
    result=raw[-12:]
    print(result)
    send_mail('The result of '+model+':\n'+str(result))



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
        
    

