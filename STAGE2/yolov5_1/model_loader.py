import torch
from pathlib import Path, WindowsPath
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
import numpy as np
from utils.general import check_dataset, check_img_size, colorstr, non_max_suppression
from yolov5_1.models.common import DetectMultiBackend  # Assurez-vous que ce module est correctement importé

def load_models(weights_path):
    # Charger le modèle avec DetectMultiBackend
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights_path, device=device)
    return model

def detect(model, img):
    # Utiliser le modèle pour faire la détection
    results = model(img)
    result_nms = non_max_suppression(results)
    # Extraire les résultats de détection sous forme de liste de dictionnaires
    detections = []

    for result in result_nms[0]:  # results[0] contient les résultats des détections
        print(result)
        print(result.shape)
        x1, y1, x2, y2, conf, cls = result[:6]
        detections.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "conf": float(conf),
            "cls": int(cls)
        })

    return detections