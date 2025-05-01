#!/bin/bash

CONFIG=".\yolo_config.yaml"
TRAIN_DATA=".\yolo\data\train"
VAL_DATA=".\yolo\data\val"

WEIGHTS=".\yolo\best.pt"

# 모델 학습
echo "+++식재료 yolo 학습 시작+++"
python main.py --mode train --config $CONFIG --train_path $TRAIN_DATA --val_path $VAL_DATA

