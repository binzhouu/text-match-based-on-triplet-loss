# Triplet Loss
![triplet_loss](https://github.com/binzhouu/text-match-based-on-triplet-loss/tree/master/png)





#### 模型初始化：
tm = TextMatch(ip_port='localhost:50051')
- 参数说明：  
    ip_port:faiss server的地址及端口号
#### 模型推理：
confidence, similar_text, pred_label = tm.inference(text)  
- 参数说明：  
    text:用户的input_text 类型：str  
    return:  
       confidence:置信度 类型：float  
       similar_text:top1接近的text 类型：str  
       pred_label:预测结果标签 类型：str