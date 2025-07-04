from ultralytics import YOLO

model = YOLO("C:/Users/amir/Desktop/Zebra/runs/detect/train10/weights/best.pt")

vertical_imgsz = (640, 1280)

model.predict(source= "3_1.MOV", show=True, save=True, conf=0.7, line_width = 2, imgsz = vertical_imgsz,  save_crop = False, save_txt = False, show_labels = True, show_conf = True, classes = [0,1,2,3,4])