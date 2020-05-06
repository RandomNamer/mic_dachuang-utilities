import smtplib
from email.mime.text import MIMEText

msg=MIMEText('Result of epoch 1:\n'+str([' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.148\n', ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.294\n', ' Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.135\n', ' Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050\n', ' Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.164\n', ' Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.093\n', ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.048\n', ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.293\n', ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.466\n', ' Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.250\n', ' Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473\n', ' Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423\n']),'plain','utf-8')

usrn='smtpservicetest@163.com'
pswd='BBSKRKQDYQFJIXKG'
sendto='random_name@sjtu.edu.cn'

msg['From']='smtpservice'
msg['To']=sendto
msg['Subject']='Training breif'


smtp=smtplib.SMTP()
smtp.set_debuglevel(1)
smtp.connect('smtp.163.com',25)
smtp.login(usrn,pswd)
smtp.sendmail(usrn,[sendto],msg.as_string())
