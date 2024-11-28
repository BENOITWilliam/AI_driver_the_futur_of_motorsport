import shutil
from pathlib import Path
import logging
import cv2
import yaml
from ultralytics import YOLO

# Configuration des logs
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['names']

def process_images(model_version_name):
    # Charger le modèle avec les meilleurs poids
    model = YOLO(f'./runs/detect/yolo11s_color_test_/weights/best.pt')
    
    # Charger les noms des classes à partir du fichier YAML
    yaml_path = './dataset/data.yaml'
    class_names = load_class_names(yaml_path)
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path('test/dataset_test_output')
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parcourir les images dans le dossier d'entrée
    input_dir = Path('dataset/data/test/images')
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Lire l'image en couleur
            img_color = cv2.imread(str(img_path))
            
            results = model.predict(source=img_color, conf=0.25)
            
            # Dessiner les cases et les labels sur l'image en couleur originale
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label_index = int(box.cls[0])  # Assurez-vous que le label est correctement récupéré
                    confidence = box.conf[0]  # Assurez-vous que la confiance est correctement récupérée
                    label = class_names[label_index]
                    text = f'{label}: {confidence:.2f}'
                    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_color, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Sauvegarder l'image annotée
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), img_color)
            
            logging.info(f"Traité : {img_path.name}")


#! Launch zone : modify the model_version_name variable to the desired model version name
if __name__ == '__main__':
    model_version_name = 'yolo11s_color_test_'  # version name : yolo11m_grayscale_test_ or yolo11s_color_test_ or yolo11s_grayscale_test_
    process_images(model_version_name)