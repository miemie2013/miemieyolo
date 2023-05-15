pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple



wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s6.pt



改动：
加上支持coco json格式数据集, 即
force_coco_json: True



----------------------- eval -----------------------
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val --reproduce_640_eval --device 3


 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.618
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.489
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.708
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.814








python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s6.pt --task val --reproduce_640_eval --img 1280 --device 3





----------------------- 迁移学习 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &

1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth     > ppyolo.log 2>&1 &



python tools/train.py --batch 8 --epochs 2 --eval-interval 1 --workers 2 --conf configs/yolov6s_finetune.py --data data/voc2012.yaml --fuse_ab --device 3

nohup python tools/train.py --batch 8 --epochs 16 --eval-interval 4 --workers 2 --conf configs/yolov6s_finetune.py --data data/voc2012.yaml --fuse_ab --device 3     > yolov6s_finetune.log 2>&1 &




- - - - - - - - - - - - - - - - - - - - - -

