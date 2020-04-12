## 场景匹配：

### 路径：models/text_match_v1/text_match_for_scene.py

- 参数说明：  
    text: 用户的input_text;  
    text_list: 自定义构建的场景相关语料，用于匹配  
    label_list: 自定义构建的场景相关语料，所对应的标签（注：标签都采用英文）
    model:当前加载的模型
   
- 示例：
    #### 初始化模型
        dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
        model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')
        data = Data(ori_texts=[], labels=[], if_train=False)
        with open(dset_path, 'rb') as rbf:
            data.char_alphabet.instance2index = pickle.load(rbf)
            data.word_alphabet.instance2index = pickle.load(rbf)
            data.label_alphabet.instance2index = pickle.load(rbf)
            data.char_alphabet_size = pickle.load(rbf)
            data.word_alphabet_size = pickle.load(rbf)
            data.label_alphabet_size = pickle.load(rbf)
            data.label_alphabet.instances = pickle.load(rbf)
            data.train_texts = pickle.load(rbf)
        data.fix_alphabet()
        model = TextMatchModel(data)
        model.load_state_dict(torch.load(model_dir, map_location=model.configs['map_location']))
        model.eval()
        model.to(model.configs['device'])
    #### 自定义场景相关语料及标签
        scene_name = []
        text_list, label_list = [], []
        scenes = [{'name': '回家'}, {'name': '休息'}]
        for scene in scenes:
            scene_name.append(scene['name'])
            text_list.append(scene['name'])  # add scene name
            label_list.append('start_scene')
            # 扩展话术:
            text_list.append('打开' + scene['name'] + '场景')  # add scene name
            label_list.append('start_scene')
            text_list.append('开启' + scene['name'] + '场景')  # add scene name
            label_list.append('start_scene')
            text_list.append('打开' + scene['name'] + '模式')  # add scene name
            label_list.append('start_scene')
            text_list.append('开启' + scene['name'] + '模式')  # add scene name
            label_list.append('start_scene')
            text_list.append('关闭' + scene['name'] + '模式')
            label_list.append('close_scene')
        texts = ['我回家了啊', '关闭回家把', '开启回家', '打开回家吧', '开回家场景', '打开休息模式吧']  
	#### 模型推理  
        for text in texts:
        confidence, similar_text, pred_label = model.inference_for_scene(text, text_list, label_list, model)
        print(confidence, similar_text, pred_label)  

## 意图匹配：

### 路径：models/text_match_v1/text_match_for_scene.py

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