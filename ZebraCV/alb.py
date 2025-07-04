import albumentations as A
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


INPUT_IMAGE_DIR = "C:/Users/amir/Desktop/alb/images"          
INPUT_LABEL_DIR = "C:/Users/amir/Desktop/alb/labels"          
OUTPUT_IMAGE_DIR = "C:/Users/amir/Desktop/alb/aug_img"     
OUTPUT_LABEL_DIR = "C:/Users/amir/Desktop/alb/aug_lab"     
AUG_PER_IMAGE = 5                    
VISUALIZE_RESULTS = True             



os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.2, p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),          
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=0.3),  
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),  
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
], bbox_params=A.BboxParams(
    format='yolo',
    min_visibility=0.2,
    min_area=32,
    label_fields=['class_ids']
))

def visualize_bboxes(img, bboxes, title="Image"):
    """Визуализация bounding box на изображении"""
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()
    
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x_min = (x_center - width/2) * img.shape[1]
        y_min = (y_center - height/2) * img.shape[0]
        rect = plt.Rectangle(
            (x_min, y_min),
            width * img.shape[1],
            height * img.shape[0],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


total_aug = 0
image_paths = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.png"))

for img_path in image_paths:
    
    image = cv2.imread(img_path)
    if image is None:
        continue
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    
    
    label_path = os.path.join(INPUT_LABEL_DIR, os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
    bboxes = []
    class_ids = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    bboxes.append([x_center, y_center, width, height])
                    class_ids.append(class_id)
    
    
    for aug_idx in range(AUG_PER_IMAGE):
        try:
            
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_ids=class_ids
            )
            
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_ids = transformed['class_ids']
            
            
            base_name = os.path.basename(img_path).rsplit('.', 1)[0]
            output_img_name = f"{base_name}_aug{aug_idx+1}.jpg"
            output_txt_name = f"{base_name}_aug{aug_idx+1}.txt"
            
            
            output_img_path = os.path.join(OUTPUT_IMAGE_DIR, output_img_name)
            cv2.imwrite(output_img_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            
            
            output_label_path = os.path.join(OUTPUT_LABEL_DIR, output_txt_name)
            with open(output_label_path, 'w') as f:
                for bbox, class_id in zip(transformed_bboxes, transformed_class_ids):
                    f.write(f"{int(class_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            total_aug += 1
            
            # Визуализация первого результата для каждого изображения
            if VISUALIZE_RESULTS and aug_idx == 0:
                print(f"Original: {img_path} | Boxes: {len(bboxes)}")
                print(f"Augmented: {output_img_name} | Boxes: {len(transformed_bboxes)}")
                visualize_bboxes(image, bboxes, "Original Image")
                visualize_bboxes(transformed_image, transformed_bboxes, "Augmented Image")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

print(f"\nАугментация завершена! Сгенерировано {total_aug} новых образцов")
print(f"Изображения сохранены в: {os.path.abspath(OUTPUT_IMAGE_DIR)}")
print(f"Аннотации сохранены в: {os.path.abspath(OUTPUT_LABEL_DIR)}")