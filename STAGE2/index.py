from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
from PIL import Image
from io import BytesIO
import base64
import os
import numpy as np
import torch
from pathlib import Path, WindowsPath
from ajouter import ajouter, verif_valid
from connecter import verif_conn
from YoloBD import *
from img_bd import *
import sys
import yaml
from yaml.loader import SafeLoader
# Ajoutez le chemin du dossier yolov5_1 au sys.path
sys.path.insert(1, 'C:/Users/remyg/Documents/STAGE2/yolov5_1')
from yolov5_1.model_loader import load_models, detect

app = Flask(__name__)
app.secret_key = "192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf"

UPLOAD_FOLDER = WindowsPath(r'C:\Users\remyg\OneDrive\Documents\L3\STAGE\static\img_load')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['POST', 'GET'])
def hello():
    image_id, image_url = get_random_image_from_db(3)
    if image_url:
        image1, image2, image3 = image_url
        if 'username' in session:
            return render_template('home.html', image_1=image1, image_2=image2, image_3=image3, image_id=image_id)
        else:
            return render_template('home_no_conn.html', image_1=image1, image_2=image2, image_3=image3, image_id=image_id)
    else:
        return "Erreur : impossible de récupérer l'image depuis la base de données"


@app.route('/detect', methods=['POST'])
def detect_img():
    if request.method == 'POST':
        weights_path = 'C:/Users/remyg/Documents/STAGE2/yolov5_1/best.pt'
        model_name = "yolov5_1"
        model = load_models(weights_path)

        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return "Aucun fichier sélectionné"

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Charger l'image avec PIL et la convertir en ndarray OpenCV
            img = Image.open(image_file.stream).convert("RGB")
            img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Redimensionner l'image comme dans votre script
            height, width, _ = img_cv2.shape
            img_resized = cv2.resize(img_cv2, (640, 640))

            # Préparer l'image pour YOLOv5
            img_tensor = torch.from_numpy(img_resized).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0

            # Effectuer la détection avec YOLOv5
            detections = detect(model, img_tensor)

            # Debug: Print detections before replacing class IDs with names
            print("Detections before replacing class IDs:", detections)

            # Remplacer les classes par leurs noms
            data_path = os.path.join('C:/Users/remyg/Documents/STAGE2', model_name, "data/data.yaml")
            raw, names = import_lab(data_path)
            print("Class names loaded:", names)

            for detection in detections:
                cls_id = detection.get('cls')
                if cls_id is not None:
                    detection['name'] = names.get(cls_id, 'Unknown')

            # Debug: Print detections after replacing class IDs with names
            print("Detections after replacing class IDs:", detections)

            # Récupérer l'image avec les boîtes de détection
            img_with_boxes = img_resized.copy()
            for det in detections:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                conf = det['conf']
                class_name = det['name']
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 87, 51), 2)  # Boîtes de couleur orange
                cv2.putText(img_with_boxes, f'{conf:.2f} - {class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 87, 51), 2)

            # Convertir l'image en bytes
            img_byte_array = BytesIO()
            Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)).save(img_byte_array, format='JPEG')
            img_data = img_byte_array.getvalue()

            # Encoder les données d'image en base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            image_url = url_for('static', filename='img_load/' + image_file.filename)

            # Vérifier si l'image doit être enregistrée
            if should_save_image(detections, threshold_confidence=0.5):
                nom = session['username']
                if ajt_img(image_url, nom):
                    print("<script>alert('img!'); window.location.href='/';</script>")
                else:
                    print("<script>alert('not!'); window.location.href='/';</script>")

            # Retourner le template result.html avec les données de détection
            return render_template('result.html', detections=detections, image=img_base64, img_url=image_url)

        else:
            return "Aucun fichier trouvé dans la requête"

# Fonctions utilitaires adaptées

def import_lab(path):
    try:
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=SafeLoader)
        raw = data.get("names", [])  # Utilisation de .get() pour éviter KeyError si 'names' n'est pas présent
        names = {i: name for i, name in enumerate(raw)}
        return raw, names
    except Exception as e:
        print(f"Error loading names from {path}: {e}")
        return [], {}
    
def should_save_image(detections, threshold_confidence=0.5):
    # Vérifiez si la liste de détections n'est pas vide
    if detections:
        # Trouver la confiance maximale parmi les détections
        max_confidence = max(d['conf'] for d in detections)
        return max_confidence >= threshold_confidence
    return False  # Retourner False si la liste de détections est vide



@app.route('/login')
def login():
    return render_template('connexion.html')

@app.route('/register')
def register():
    return render_template('inscription.html')

@app.route('/auth', methods=['POST', 'GET'])
def auth():
    if request.method == 'POST':
        donnees = request.form
        nom = donnees.get('username')
        email = donnees.get('email')
        mdp = donnees.get('password')
        exp = donnees.get('Check1') is not None  # Convert to boolean
        if verif_valid(nom, email):
            return "<script>alert('Username ou email déjà utilisé!'); window.location.href='/register';</script>"
        else:
            ajouter(nom, email, mdp, exp)
            return redirect(url_for('hello'))

@app.route('/conn', methods=['POST', 'GET'])
def conn():
    if request.method == 'POST':
        donnees = request.form
        nom = donnees.get('username')
        mdp = donnees.get('password')
        valid, is_expert = verif_conn(nom, mdp)
        if valid:
            session['username'] = nom
            session['is_expert'] = is_expert  # Store the expert status in the session
            return redirect(url_for('hello'))
        else:
            return "<script>alert('Username ou mot de passe incorrect!'); window.location.href='/login';</script>"

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('is_expert', None)
    return redirect(url_for('hello'))

@app.route('/bd_activ', methods=['POST', 'GET'])
def bd_activ():
    if request.method == 'POST':  
        donnees = request.form
        nom = session['username']
        comment = donnees.get('comment')
        img_url = donnees.get('image_data')
        activ = np.random.randint(0, 3)
        if ajt_comment(img_url, comment, nom, activ):
            return "<script>alert('commentaire enregistré!'); window.location.href='/';</script>"
        else:
            return "<script>alert('not select!'); window.location.href='/';</script>"

@app.route('/Activ')
def random_image():
    if 'is_expert' in session and session['is_expert']:
        image_ids, image_urls = get_random_image_from_db(1)
        if image_urls:
            return render_template('activ.html', image_url=image_urls, image_id=image_ids)
        else:
            return "Erreur : impossible de récupérer l'image depuis la base de données"
    else:
        return redirect(url_for('no_expert'))

@app.route('/no_expert')
def no_expert():
    return render_template('no_expert.html')

@app.route('/save_zones', methods=['POST'])
def save_zones():
    zones_data = request.json  # Récupérer les données des zones depuis la requête JSON
    data = ""
    for zone in zones_data:
        x = zone['x']
        y = zone['y']
        width = zone['width']
        height = zone['height']
        label = zone['label']
        img_id = zone['id']
        if label != "":
            data += f"{x}/{y}/{width}/{height}/{label}" + "|||"
    ajt_label(img_id, data)
    get_annotations(img_id)
    
    # Example usage
    source_directory = 'C:/Users/remyg/Documents/STAGE2/static/img_load'
    annotations_directory = 'C:/Users/remyg/Documents/STAGE2/yolov5/annotations'
    train_directory = 'C:/Users/remyg/Documents/STAGE2/yolov5/data/image/train'
    val_directory = 'C:/Users/remyg/Documents/STAGE2/yolov5/data/image/val'
    test_directory = 'C:/Users/remyg/Documents/STAGE2/yolov5/data/image/test'

    split_dataset(source_directory, annotations_directory, train_directory, val_directory, test_directory)
    return jsonify({'message': 'Données des zones enregistrées avec succès'})



if __name__ == '__main__':
    app.run(debug=True)
