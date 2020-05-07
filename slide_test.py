# -*- coding: utf-8 -*-
"""
@Author :       wyl
@Email :  wangyl306@163.com
@Time  :   2020/2/22 12:58
@Project : 11-安全带
@FileName:  mytest.py
"""
from mmdet.apis import init_detector, inference_detector, show_result
import os
import time
import cv2

config_file = 'new_configs/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth'
test_path="data/ok-2"
save_path = "data/res_ok/"
(winW,winH)=(512,512)
stepSize=(512,512)

print("config_file:",config_file)
print("checkpoint_file :",checkpoint_file)
print("test_paths :",test_path)
print("pred.....................")
# or save the visualization results to image file
if not os.path.exists(save_path):
    os.makedirs(save_path)
def sliding_window(image,stepSize,windowsSize):
    for y in range(0,image.shape[0],stepSize[1]):
        for x in range(0,image.shape[1],stepSize[0]):
            yield (x,y,image[y:y+windowsSize[1],x:x+windowsSize[0]])
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

for file in os.listdir(test_path):
    # test a single image and show the results
    img = os.path.join(test_path,file)  # or img = mmcv.imread(img), which will only load it once
    img=cv2.imread(img)
    cnt=0
    myresult=[]
    start = time.time()
    for (x,y,window) in sliding_window(img,stepSize=stepSize,windowsSize=(winW,winH)):
        if window.shape[0] != winH or window.shape[1] !=winW:
            continue
        slice=img[y:y+winH,x:x+winW]
        cnt+=1
        result = inference_detector(model, slice)
        # print(result)
        for i in range(len(result)):
            if result[i].size != 0 and result[i][0,4] >= 0.2:
                label=model.CLASSES[i]
                x1=result[i][0, 0] + x
                y1=result[i][0, 1] + y
                x2=result[i][0, 2] + x
                y2=result[i][0, 3] + y
                res=[x1,y1,x2,y2,label]
                myresult.append(res)
    # print(myresult)
    end = time.time()
    print("ETA:", end - start)
    for r in myresult:
        # print(r)
        cv2.rectangle(img,(int(r[0]),int(r[1])),(int(r[2]),int(r[3])),(0,0,255),4)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,r[4],(int(r[0]-2),int(r[1])-2),font,1,(0,255,0),1)
    cv2.imwrite(save_path+file.split(".")[0]+'.jpg',img)
    #     for i in range(len(result)):
    #         if result[i].size !=0 and result[i][0,4] >= 0.3:
    #             result[i][0, 0] += x
    #             result[i][0, 1] += y
    #             result[i][0, 2] += x
    #             result[i][0, 3] += y
    #             show_result(img, result, model.CLASSES, out_file=save_path+file.split(".")[0]+"_"+str(cnt)+'.jpg')
print("pred_over,save result_train in :",save_path)