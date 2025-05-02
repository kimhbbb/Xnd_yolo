#!/bin/bash

CONFIG="yolo_config.yaml"
TRAIN_DATA="data/merged/train"
VAL_DATA="data/merged/val"

WEIGHTS="output/best.pt"

# 모델 학습
echo "+++식재료 yolo 학습 시작+++"
python main.py --mode train --config $CONFIG

