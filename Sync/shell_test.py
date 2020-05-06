import os

str=os.popen('echo fuck').readlines()
print(str)