import torch
from ultralytics import YOLO
import argparse
from detector import IngredientDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='식재료 탐지')
    parser.add_argument('--config', type=str, default='./yolo_config.yaml', help='설정 파일 경로')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train','val','test'], help='실행 모드')
    parser.add_argument('--weights', type=str, default=None, help='학습 파람 파일 경로')
    parser.add_argument('--train_path', type=str, default=None, help='train 데이터 경로')
    parser.add_argument('--val_path', type=str, default=None, help='val 데이터 경로')
    parser.add_argument('--input', type=str, default=None, help='input 경로')
    parser.add_argument('--output', type=str, default=None, help='output 경로')

    args = parser.parse_args()

    detector = IngredientDetector(config_path=args.config)

    if args.mode == 'train':
        detector.train()
    elif args.mode == 'val':
        detector.validate(weights = args.weights)
    elif args.mode == 'test':
        if args.input:
            detector.test(args.input, weights = args.weights)