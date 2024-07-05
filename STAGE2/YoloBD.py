import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from BD import get_BD
import mysql.connector
from PIL import Image
import shutil
import random

def save_annotations_to_file(annotations, img_name, image_width, image_height):
    def convert_to_yolo_format(x, y, width, height, img_width, img_height):
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        normalized_width = width / img_width
        normalized_height = height / img_height
        return x_center, y_center, normalized_width, normalized_height

    def get_class_id(label):
        class_mapping = {
            'person': 0,
            'car': 1,
            'bicycle': 2,
            # Add other mappings here
        }
        return class_mapping.get(label, -1)

    # Ouvrir le fichier en mode ajout
    with open(f'yolov5/annotations/{img_name}.txt', 'a') as f:
        for annotation in annotations:
            # Vérifier que l'annotation contient les 5 valeurs attendues
            annotation_values = annotation.split('/')
            if len(annotation_values) != 5:
                print(f"Warning: Annotation '{annotation}' does not contain 5 values.")
                continue

            x, y, width, height, label = annotation_values
            x_center, y_center, normalized_width, normalized_height = convert_to_yolo_format(
                float(x), float(y), float(width), float(height), image_width, image_height
            )
            class_id = get_class_id(label)
            if class_id == -1:
                print(f"Warning: Label '{label}' not found in class mapping.")
                continue
            annotation_line = f"{class_id} {x_center} {y_center} {normalized_width} {normalized_height}\n"
            f.write(annotation_line)


def get_label_data(img_id):
    conn = get_BD()
    if conn is None:
        print("Failed to connect to the database.")
        return None, None

    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT label, img_url FROM img WHERE img_id = %s"
        cursor.execute(query, (img_id,))
        result = cursor.fetchone()
        if result:
            return result['label'], result['img_url']
        else:
            return None, None
    except mysql.connector.Error as e:
        print(f"Error executing query: {e}")
        return None, None
    finally:
        cursor.close()
        conn.close()


def get_image_dimensions(url):
    # Suppose that the image is stored locally
    image_path = f'static/img_load/{os.path.basename(url)}'
    with Image.open(image_path) as img:
        return img.width, img.height
    
def get_annotations(img_id):
    label, img_url = get_label_data(img_id)
    
    if label is None or img_url is None:
        return jsonify({'message': 'No annotations found for this image'}), 404

    # Process the annotations and image URL
    image_width, image_height = get_image_dimensions(img_url)
    annotations = label.split('|||')

    # Get image name without extension
    img_name = os.path.splitext(os.path.basename(img_url))[0]

    save_annotations_to_file(annotations, img_name, image_width, image_height)
    
    return jsonify({'message': 'Annotations saved in YOLO format', 'file_path': f'yolov5/annotations/{img_name}.txt'})



def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_dataset(source_dir, annotations_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    train_cutoff = int(len(images) * train_ratio)
    val_cutoff = train_cutoff + int(len(images) * val_ratio)

    train_images = images[:train_cutoff]
    val_images = images[train_cutoff:val_cutoff]
    test_images = images[val_cutoff:]

    # Ensure directories exist
    ensure_directory_exists(train_dir)
    ensure_directory_exists(val_dir)
    ensure_directory_exists(test_dir)

    for img in train_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))
        txt_file = img.replace('.jpg', '.txt')
        src_txt_path = os.path.join(annotations_dir, txt_file)
        dst_txt_path = os.path.join(train_dir, txt_file)
        if os.path.exists(src_txt_path):
            shutil.copy(src_txt_path, dst_txt_path)
            print(f"Copied {src_txt_path} to {dst_txt_path}")

    for img in val_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(val_dir, img))
        txt_file = img.replace('.jpg', '.txt')
        src_txt_path = os.path.join(annotations_dir, txt_file)
        dst_txt_path = os.path.join(val_dir, txt_file)
        if os.path.exists(src_txt_path):
            shutil.copy(src_txt_path, dst_txt_path)
            print(f"Copied {src_txt_path} to {dst_txt_path}")

    for img in test_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(test_dir, img))
        txt_file = img.replace('.jpg', '.txt')
        src_txt_path = os.path.join(annotations_dir, txt_file)
        dst_txt_path = os.path.join(test_dir, txt_file)
        if os.path.exists(src_txt_path):
            shutil.copy(src_txt_path, dst_txt_path)
            print(f"Copied {src_txt_path} to {dst_txt_path}")

