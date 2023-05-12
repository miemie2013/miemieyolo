pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple



wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s6.pt


----------------------- eval -----------------------
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val --reproduce_640_eval

python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s6.pt --task val --reproduce_640_eval --img 1280





----------------------- 迁移学习 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &

1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth     > ppyolo.log 2>&1 &



python tools/train.py --batch 8 --conf configs/yolov6s_finetune.py --data data/voc2012.yaml --fuse_ab --device 0




- - - - - - - - - - - - - - - - - - - - - -

