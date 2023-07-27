简体中文 | [English](README_EN.md)

# DeepLearning-based-Feature-extraction-and-matching
基于深度学习的特征点提取喝匹配算法C#推理

 ## 深度学习特征检测匹配算法</h2>  
深度学习特征提取算法：SuperPoint，以及深度学习特征匹配算法：LightGlue。  
SuperPoint：​  
[[`Paper`](https://arxiv.org/pdf/1712.07629.pdf)] [[`源码`](https://github.com/rpautrat/SuperPoint )]  

​
LightGlue:  
[[`Paper`](https://arxiv.org/pdf/2306.13643.pdf )] [[`源码`](https://github.com/cvg/LightGlue)]  


 ## 应用：基于superpoint和lightglue进行图像匹配</h2>  
1.将ONNX格式与训练模型放到exe路径下  
2.运行LightGlue.exe ，选择图像文件  
3.点击Feature按钮进行特征提取，特征点置信度越高，越接近黄色，置信度越低越接近黑色  
4.可以过滤掉置信度低的特征点  
5.点击Match进行特征点匹配  

<img width="800" src="https://user-images.githubusercontent.com/18625471/256469113-d31ae3c1-13df-4c16-ad71-359e6bd8520b.jpg">  
 


