import torch
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml

class IngredientDetector():
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        os.makedirs(self.cfg['output_dir'], exist_ok=True)

        self.model = YOLO(self.cfg['model_type'])

    def train(self):
        results = self.model.train(
            data = self.cfg['data_yaml'],
            epochs = self.cfg['epochs'],
            batch = self.cfg['batch_size'],
            imgsz = self.cfg['img_size'],
            device = self.cfg['device'],
            lr = self.cfg['lr'],
            patience = self.cfg['patience'],
            project = self.cfg['output_dir'],
            name = 'train'
        )

        print(f"학습 결과 저장 경로: {self.model.trainer.save_dir}")
        return results

    def validate(self, weights):
        

        pass

    def test(self, image_path, weights):

        pass
    