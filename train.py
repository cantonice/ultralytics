import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    # model = YOLO(yaml) # 使用这一行 就是直接加载yaml文件训练
    model = YOLO(weights)  # 使用这一行 就是直接加载权重文件进行训练
    # model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练

    model.info()

    results = model.train(data='glass_substrate.yaml',  # 训练参数均可以重新设置
                        epochs=20, 
                        imgsz=640, 
                        workers=8, 
                        batch=8,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/26/yolo26.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolo26n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)