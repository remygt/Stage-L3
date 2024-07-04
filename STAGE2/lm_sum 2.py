import pathlib
# Redéfinir PosixPath pour utiliser WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

# Ensuite, le reste de votre code fonctionne comme prévu
import os
import numpy as np
import sys
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm
from math import log2

sys.path.insert(1, 'yolov5 1')
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr, non_max_suppression
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

def import_tg(path, height, width):
    labels = []
    with open(path, 'r') as f:
        for l in f:
            ob = l.split()
            ob_int = [float(el) for el in ob]
            ob_final = [ob_int[0], (ob_int[1] - ob_int[3] / 2) * width, (ob_int[2] - ob_int[4] / 2) * height, (ob_int[1] + ob_int[3] / 2) * width, (ob_int[2] + ob_int[4] / 2) * height]
            labels.append(ob_final)
    labels = np.array(labels)
    return labels

def replace_extension(file_path, new_extension):
    base_path, old_extension = os.path.splitext(file_path)
    new_file_path = base_path + new_extension
    return new_file_path

def info_model(file):
    with open(file) as f:
        data = yaml.load(f, Loader=SafeLoader)
    _data = data["data"]
    imgsz = data["imgsz"]
    return _data, imgsz

def import_lab(path):
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    raw = data["names"]
    names = {n: i for i, n in enumerate(raw)}
    return raw, names

name = "BISES"
model_name = f"finetune_{name}"
name_result = f"lm_sum_{name}"
topm = 55

# Update the path to the dataset and weights
path = os.path.join("C:/Users/remyg/Documents/STAGE2/yolov5/")
weights = os.path.join(path, 'runs/weights/best.pt')
device = "cpu"  # Use CPU instead of CUDA
device = select_device(device, batch_size=1)
data = check_dataset(os.path.join(path, "data/data.yaml"))
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
raw, names = import_lab(os.path.join(path, "data/data.yaml"))

path_data, img_size = info_model(os.path.join(path, "runs/opt.yaml"))
batch_size = 1
task = 'val'
stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
imgsz = check_img_size(img_size, s=stride)
pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)
task = task if task in ('train', 'val', 'test') else 'val'
dataloader = create_dataloader(data[task], imgsz, batch_size, stride, False, pad=pad, rect=rect, workers=8, prefix=colorstr(f'{task}: '))[0]

filenames_val = []
valid = []
for (im, targets, path, shapes) in tqdm(dataloader):
    filenames_val.append(os.path.basename(path[0]))

    im = im.to(device, non_blocking=True)
    im = im.half() if False else im.float()
    im /= 255

    p = model(im)
    p = non_max_suppression(p)
    p = np.array([array[6:-2].cpu() for array in p[0]])
    if len(p) == 0:
        valid.append(0)
        continue

    sorted_probs = np.sort(p, axis=1)[:, ::-1]

    least_margins = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])

    summ = np.sum(least_margins)
    valid.append(summ)

def top_indices(lst, m):
    indexed_list = [(val, idx) for idx, val in enumerate(lst)]
    sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
    top_indices = [idx for val, idx in sorted_list[:m]]

    top = []
    for i in range(len(lst)):
        top.append(-1 if i in top_indices else 1)

    return top

results = top_indices(np.array(valid), topm)

try:
    os.makedirs(os.path.join("results", name_result), exist_ok=True)
except FileExistsError:
    pass

text = "Column 1\tColumn 2\n"
for a, b in zip(filenames_val, results):
    text += f"{a}\t{b}\n"

with open(os.path.join("results", name_result, "outputval.txt"), 'w') as file:
    file.write(text)

print(name_result)
