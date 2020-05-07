1. docker容器制作

```python
docker run -it --gpus all -e DISPLAY=$DISPLAY --privileged \
--name=server_$USER \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
dl_server bash

如果哪一天能使用图形界面的容器，不能映射出相应的图形程序的话
#你需要在宿主主机上面查询当前的$DISPLAY的值，然后将这个值在容器内部设置为环境变量
#宿主主机敲击命令
echo $DISPLAY
#得出一下结果
:0
#那么你在容器内布需要设置环境变量命令为,如果这台主机单个人使用的这个值是不会变的
#如果是多个人使用的，比如我们的大服务器，那么我们可能需要保持宿主主机和容易默认同一个
#容器内部敲击命令
export DISPLAY=:0
#设置宿主主机接收容器内部图形信号，宿主主机敲击命令
xhost +
```

2. 基础硬件环境

```
cuda 10.1
cudnn 7
```

3. 制作VOC数据集

```
VOCdevkit
    --VOC2007
        ----Annotations
        ----ImageSets
            ------Main
        ----JEPGImages
"""自动划分训练集、测试集"""

"""自动划分训练集、测试集"""

import os
import random

trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)


num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        fval.write(name)
        ftest.write('\n')
        #if i in train:
            #fval.write(name)
        #else:
            #ftest.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```

4. 打开 /mmdetection/mmdet/core/evaluation/class_names.py 文件，修改 voc_classes 为将要训练的数据集的类别名称

```
def voc_classes():
    return [
        "hg_ok","hg_ng","hw_ok","hw_ng"
    ]
```

5. 打开 mmdetection/mmdet/datasets/voc.py 文件，修改 VOCDataset 的 CLASSES 为将要训练的数据集的类别名称。

```
 #如果只有一个类，要加上一个逗号，否则将会报错。
 CLASSES = ("hg_ok","hg_ng","hw_ok","hw_ng")
```

6. 修改配置文件

```
将 num_classes 变量改为：类别数 + 1
```

```python
#data_setting
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
```

```
#多尺度训练

train_pipeline中dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)的keep_ratio解析。假设原始图像大小为（1500， 1000），ratio=长边/短边 = 1.5。
1. 当keep_ratio=True时，img_scale的多尺度最多为两个。假设多尺度为[(2000, 1200), (1333, 800)]，则代表的含义为：首先将图像的短边固定到800到1200范围中的某一个数值假设为1100，那么对应的长边应该是短边的ratio=1.5倍为1110x1.5=1650 ，且长边的取值在1333到2000的范围之内。如果大于2000按照2000计算，小于1300按照1300计算。

2. 当keep_ratio=False时，img_scale的多尺度可以为任意多个。假设多尺度为[(2000, 1200), (1666, 1000),(1333, 800)]，则代表的含义为：随机从三个尺度中选取一个作为图像的尺寸进行训练。

test_pipeline 中img_scale的尺度可以为任意多个，含义为对测试集进行多尺度测试（可以理解为TTA）
```

```python
#lr
#对学习率的调整，一般遵循下面的习惯： lr = 0.00125*batch_size（ = gpu_num(训练使用gpu数量) * imgs_per_gpu）
#8 gpus、imgs_per_gpu = 2：lr = 0.02；
#2 gpus、imgs_per_gpu = 2 或 4 gpus、imgs_per_gpu = 1：lr = 0.005；
#4 gpus、imgs_per_gpu = 2：lr = 0.01
#1 gpu、imgs_per_gpu = 2：lr = 0.00125 / 0.0025   
```

```python
#这里说一下 epoch 的选择，默认 total_epoch = 12，learning_policy 中，step = [8,11]。total_peoch 可以自行修改，若 total_epoch = 50，则 learning_policy 中，step 也相应修改，例如 step = [38,48]
```

7. 训练

```
CUDA_VISIBLE_DEVICES=0,1 python3 ./tools/train.py ./configs/faster_rcnn_r50_fpn_1x.py --gpus 2
```

8. 测试

+ 计算map

```
python tools/test.py configs/faster_rcnn_r50_fpn_1x_voc.py \
    checkpoints/SOME_CHECKPOINT.pth \
    --eval mAP
```

**预训练权重**

mmdet默认加载权重优先级别是resume_from(断点加载)，load_from，pretrained的顺序

```python
#修改类别的预训练权重
def main():
    #gen coco pretrained weight
    import torch
    num_classes = 21
    model_coco = torch.load("cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth")

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][:num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][:num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][:num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]
    # save new model
    torch.save(model_coco, "cascade_rcnn_r50_coco_pretrained_weights_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    main()
```

 **引入albumentations数据增强库进行增强**

​	mmdetection_v1.0.0已经支持

```python
#使用具体修改方式如下：添加dict(type='Albu', transforms = [{"type": 'RandomRotate90'}])，其他的类似。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Albu', transforms = [
        {"type": 'RandomRotate90'},
        {"type":"CLAHE"}, 
        dict(type="RandomBrightnessContrast",p=0.5)]),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
```

**注意**

注意：修改mmdet里面的代码，只在自己的项目中进行修改是会报错的。以修改mmdet/datasets/pipeline的transform.py为例，解决方案如下：
只在项目文件里修改mmdet/datasets/pipeline里面的数据增强是不够的，由于训练的时候导入的路径并不是项目本身路径，在项目运行之前就已经通过setup.py文件将mmdet模块发布到了site-packages路径里面，作为一个库来调用，所以每次更新模型后，都需要使用pip install .（或手动将改动过的代码添加进去）将新的mmdet模块导入site-packages中。

**anchor比例**

```python
import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)


# 读取数据
ann_json = 'instances_train2017.json'
with open(ann_json) as f:
    ann=json.load(f)

#################################################################################################
#创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
counts_label=dict([(i['name'],0) for i in ann['categories']])
for i in ann['annotations']:
    counts_label[category_dic[i['category_id']]]+=1

# 标注长宽高比例
box_w = []
box_h = []
box_wh = []
categorys_wh = [[] for j in range(10)]
for a in ann['annotations']:
    if a['category_id'] != 0:
        box_w.append(round(a['bbox'][2],2))
        box_h.append(round(a['bbox'][3],2))
        wh = round(a['bbox'][2]/a['bbox'][3],0)
        if wh <1 :
            wh = round(a['bbox'][3]/a['bbox'][2],0)
        box_wh.append(wh)

        categorys_wh[a['category_id']-1].append(wh)


# 所有标签的长宽高比例
box_wh_unique = list(set(box_wh))
box_wh_count=[box_wh.count(i) for i in box_wh_unique]

# 绘图
wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
wh_df.plot(kind='bar',color="#55aacc")
plt.show()
```

注意：

选取时得保证一个原则：不能选择极端比例。意思就是，不是有什么比例就选择什么比例，而是用一个近似比例代替其他的比例。就好像3.0可以近似的看做2.0，4.0、6.0、7.0可以近似的看为5.0，29.0可以近似的看为10.0



为了防止你所认为的长边是短边，你所认为的短边是长边。因为你也不知道哪一个是长边，哪一个是短边呢，所以把长短边的比例取它的倒数这样就解决了。

