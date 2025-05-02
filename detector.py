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

    def _get_data_path(self):
        """
        data.yaml 파일의 경로를 정확하게 처리합니다.
        이 함수는 data.yaml 파일이 현재 작업 디렉토리에 대해 상대적인 경로로 지정되어 있을 때
        올바른 경로를 반환합니다.
        """
        data_yaml_path = self.cfg['data_yaml']
        data_yaml_dir = os.path.dirname(data_yaml_path)
        
        # 데이터 디렉토리가 없으면 생성
        if not os.path.exists(data_yaml_dir):
            os.makedirs(data_yaml_dir, exist_ok=True)
            
        # data.yaml 파일이 있는지 확인
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data file not found: {data_yaml_path}")
            
        return data_yaml_path

    def train(self):
        data_yaml_path = self._get_data_path()
        print(f"데이터 구성 파일 경로: {data_yaml_path}")

        results = self.model.train(
            data = self.cfg['data_yaml'],
            epochs = self.cfg['epochs'],
            batch = self.cfg['batch_size'],
            imgsz = self.cfg['img_size'],
            device = self.cfg['device'],
            lr0 = self.cfg['lr'],
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
    