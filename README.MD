## Triplet Loss

三元组损失，原本用于图像人脸embedding的生成，很好理解，公式如下，这里我们用来作为句子represent的训练损失，
来收敛句子的embedding：  

### 损失函数公式：
![triplet_loss](png/formula.png)

### 可视化损失:
![triplet_loss](png/triplet_loss.jpg)

### 损失函数的pytorch实现：
代码已做过gpu上训练的适配：  
[triplet_loss.py](model/triplet_loss.py)
- 实现策略：  
我们选择的是batch_hard策略，即在一个batch中，循环定义每一个样本为anchor，选择距离anchor最大的positive exmaple
记为：d(a,p)；选择距离anchor最小negtive example，记为d(a,n)，我们的目标是通过训练尽可能缩小d(a,p)，增大d(a,n)；
对于1个anchor来说，loss=max(hardest_positive_dist - hardest_negative_dist + margin,0)。
假定batch=64，距离计算采用欧式距离，我们会将一个batch中的距离求平均，作为当前batch的损失

## Model Structure 
- 我们采用字符 + 词 + 词特征的融合方式来加强语义的表征；
- 模型结构采用经典的CNN + 时序模型Bilstm


### 模型初始化：
tm = TextMatch(ip_port='localhost:50051')
- 参数说明：  
    ip_port:faiss server的地址及端口号
### 模型推理：
confidence, similar_text, pred_label = tm.inference(text)  
- 参数说明：  
    text:用户的input_text 类型：str  
    return:  
       confidence:置信度 类型：float  
       similar_text:top1接近的text 类型：str  
       pred_label:预测结果标签 类型：str