# -*- coding: utf-8 -*-
"""
@Author :       wyl
@Email :  wangyl306@163.com
@Time  :   2020/2/22 12:58
@Project : 11-安全带
@FileName:  mytest.py
"""
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os

config_file = 'new_configs/r50_mutiscale_focalloss_softnms_dcn.py'
checkpoint_file = 'work_dirs/r50_mutiscale_focalloss_softnms_dcn/latest.pth'
test_paths="data/test_data"
print("config_file:",config_file)
print("checkpoint_file :",checkpoint_file)
print("test_paths :",test_paths)
print("pred.....................")
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:2')
allDir = os.listdir(test_paths)  # 列出指定路径下的全部文件夹，以列表的方式保存
for dir in allDir:  # 遍历指定路径下的全部文件和文件夹
    test_path = os.path.join(test_paths, dir)
    for file in os.listdir(test_path):
        # test a single image and show the results
        img = os.path.join(test_path,file)  # or img = mmcv.imread(img), which will only load it once
        result = inference_detector(model, img)
        # or save the visualization results to image files
        save_path = "data/result/"+dir+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        show_result(img, result, model.CLASSES, out_file=save_path+file.split(".")[0]+'.jpg')
print("pred_over,save result in :",save_path)


