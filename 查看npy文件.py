import numpy as np
# 在训练自己神经网络的时候，经常会用到已经训练好的权重来初始化自己的网络。
# 一个权重文件，后缀名是npy

test=np.load('.\model\pts_in_hull.npy',encoding = "latin1")  #加载文件
doc = open('npy读取.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中