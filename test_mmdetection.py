from mmdet.apis import init_detector, inference_detector

# config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
config_file = '/raid/liujiachen/workspace/NJAL/dependencies/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
# checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
checkpoint_file = '/raid/liujiachen/workspace/NJAL/dependencies/mmdetection/checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:4')  # or device='cuda:0'
x = inference_detector(model, '/raid/liujiachen/workspace/NJAL/dependencies/mmdetection/demo/demo.jpg')
print(x)
