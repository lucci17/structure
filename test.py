import os  

path = "D:/deeplab/tensorflow-deeplab-v3/grass/VOCdevkit/VOC2012/SegmentationClassVisualization" #文件夹目录  
files= os.listdir(path) #得到文件夹下的所有文件名称  
s = []  
for file in files: #遍历文件夹  
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
          str = ""
          str = os.path.split(file)[-1].split('.')[0]
#          str = "grass" + str
#          os.rename(os.path.join(path,file),os.path.join(path,str +".jpg"))
          s.append(str) #每个文件的文本存到list中  

f = open('D:/deeplab/tensorflow-deeplab-v3/grass/train.txt','w')
for i in range(0,100):
    f.write(s[i]+'\n')
f.close()
f = open('D:/deeplab/tensorflow-deeplab-v3/grass/val.txt','w')
for i in range(101,142):
    f.write(s[i]+'\n')
f.close()

path = "D:/deeplab/tensorflow-deeplab-v3/grass/VOCdevkit/VOC2012/SegmentationClass" #文件夹目录  
files= os.listdir(path) #得到文件夹下的所有文件名称  
i = 0
for file in files: #遍历文件夹  
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
          str = ""
          str = os.path.split(file)[-1].split('.')[0]
          str = s[i]
          os.rename(os.path.join(path,file),os.path.join(path,str +".png"))
          i = i + 1
          s.append(str) #每个文件的文本存到list中
