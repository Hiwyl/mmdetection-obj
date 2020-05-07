from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result

# 模型配置文件
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'

# 预训练模型文件
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:2')

# 测试单张图片
img = 'demo/demo.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES,out_file='result.jpg')


