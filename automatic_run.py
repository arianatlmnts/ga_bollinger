import os

for i in range(20):
    print('----------------------------process:',i+1)
    cmd = 'time python3 main.py'
    os.system(cmd)

