from ultralytics import YOLO
import os
import torch

def main():
    yaml_path = "dataset_custom.yaml"
    print("Файл существует:", os.path.exists(yaml_path))
    print("Текущая директория:", os.getcwd())
    
    # Загрузка модели
    model = YOLO("yolo11m.pt")
    
    
    model.train(
        data= "C:/Users/amir/Desktop/Zebra/dataset_custom.yaml",
        imgsz=640,
        batch=8,
        cos_lr=True,
        epochs=100,
        label_smoothing=0.1,
        workers=1,
        device=0
    )

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()