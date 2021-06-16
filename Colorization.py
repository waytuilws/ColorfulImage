from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from cv2 import dnn  #使用DNN模块调用基于CNN的训练模型

# 绝对路径，文件夹放桌面上
prototxt = r"C:\Users\Administrator\Desktop\project\model\colorization_deploy_v2.prototxt.txt"
model = r"C:\Users\Administrator\Desktop\project\model\colorization_release_v2.caffemodel"
points = r"C:\Users\Administrator\Desktop\project\model\pts_in_hull.npy"


class Root(Tk): # Root类：继承于Tk类 
    def __init__(self):
        super(Root, self).__init__()
        self.title("灰度图像着色")
        self.minsize(300, 500)

        #初始界面上的相关信息
        self.labelFrame0 = ttk.LabelFrame(self, text ='请选择图像')
        self.labelFrame0.grid(column = 0, row = 1, padx = 5, pady = 5)
        
        self.labelFrame1 = ttk.LabelFrame(self, text ='路径')
        self.labelFrame1.grid(column = 0, row = 3, padx = 5, pady = 5)

        self.labelFrame2 = ttk.LabelFrame(self, text ='图像')
        self.labelFrame2.grid(column = 0, row = 4, padx = 5, pady = 5)

        self.labelFrame3 = ttk.LabelFrame(self, text ='运行')
        self.labelFrame3.grid(column = 0, row = 5, padx = 5, pady = 5)

        self.button() # 声明button事件函数
        
    def button(self): # 定义button事件
        self.button = ttk.Button(self.labelFrame0, text = '打开文件夹', width=50,command = self.fileDialog)
        self.button.grid(column = 0, row = 1)
        
        self.button1 = ttk.Button(self.labelFrame3, text = '为该图像着色', width=50,command = self.RunPro)
        self.button1.grid(column = 0, row = 1)

    def RunPro(self):
    
        image = self.path
        print("[提示]模型加载中...")
        net = cv2.dnn.readNetFromCaffe(prototxt,model) 
        # cv2.dnn.readNetFromCaffe(prototxt, model)  用于进行SSD网络的caffe框架的加载，
        # prototxt表示caffe网络的结构文本 即时神经层 即model文件夹中的colorization_deploy_v2.prototxt.txt
        # model是已经训练好的参数结果colorization_release_v2.caffemodel

        pts = np.load(points,encoding='bytes') # 加载权重np数组，用于初始化自己神经网络
        print("[提示]模型加载完成")
        # 将簇中心作为1x1卷积添加到模型中 
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        # 从磁盘加载图像，并将每个像素的强度缩放到范围[0，1] 
        image = cv2.imread(image)   # 读入图像为BGR格式，各通道取值范围0-255
        scaled = image.astype("float32") / 255.0    # 图片归一化操作
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)   # 将图片由BGR格式转为LAB

        resized = cv2.resize(lab, (224, 224))  # 将lab图像缩放至224 x 224，网络输入尺寸
        L = cv2.split(resized)[0]   # 将图像按通道进行拆分，取灰度层，作为模型输入
        L -= 50     # 平均减法
        '''
        layer {
            name: "data_l"
            type: "Input"
            top: "data_l"
            input_param {
                        shape { dim: 1 dim: 1 dim: 224 dim: 224 }由于prototxt第一层的参数需要224*224像素，代码66行使用resize进行修改尺寸
                    }
            }
        '''

        print("[提示]正在对图像上色...")
        net.setInput(cv2.dnn.blobFromImage(L))  # 将灰度层输入模型
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # 通过模型返回ab值,同时进行三维矩阵的轴变换

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))   # 还原ab层的尺寸为原图尺寸

        L = cv2.split(lab)[0]   # 在原图的lab格式中取出灰度层，保证其尺寸与ab相同
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        # 将L层增加一个维度，使得与ab维度相同，将其进行连接
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) # 将输出图像转换为RGB 
        colorized = np.clip(colorized, 0, 1)    # 剪切超出范围0-1的像素强度

        colorized = (255 * colorized).astype("uint8")   # 将像素强度恢复到[0,255]范围内
        cv2.imshow("Colorized", colorized)
        print('[提示]上色完成')
        cv2.waitKey(0)

    def fileDialog(self):#打开文件框函数
        self.filename = filedialog.askopenfilename(initialdir= '/',title = '选择图像', filetype = (('jpeg','*.jpg'),('All Files','*.*')))
        
        # 输入框 用来显示路径
        self.e1 = ttk.Entry(self.labelFrame1, width = 50)
        self.e1.insert(0, self.filename)
        self.e1.grid(row=2, column=0, columnspan=50)

        newpath=self.filename

        # r表示'\'不是转义字符
        self.path = newpath.replace('/',r'\\')
        print (self.path)
        
        im = Image.open(self.path)
        w, h = im.size
        print(w,h)

        # 自定义图像自适应预览（长边填充）
        if(w>h):
            resized = im.resize((300,int(h/w*300)),Image.ANTIALIAS) 
        else:
            resized = im.resize((int(w/h*300),300),Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized)
        myvar=ttk.Label(self.labelFrame2,image = tkimage)
        myvar.image = tkimage
        myvar.grid(column=0, row=4)

if __name__ == '__main__': # 主函数
    root = Root() # 建立一个root窗口对象
    root.mainloop() # 显示这个root窗口对象