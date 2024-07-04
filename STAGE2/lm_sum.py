import os 
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms
import cv2
from pred import Feature_prediction
import sys 
sys.path.insert(1, 'yolov5')
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr, non_max_suppression
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from tqdm import tqdm
import yaml 
from yaml.loader import SafeLoader
import random
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from math import log2

def import_tg(path,height,width):
    labels = []
    with open(path,'r') as f:
        for l in f:
            ob =  l.split()
            ob_int = [float(el) for el in ob]
            ob_final = [ob_int[0], (ob_int[1]-ob_int[3]/2)*width,(ob_int[2]-ob_int[4]/2)*height,(ob_int[1]+ob_int[3]/2)*width,(ob_int[2]+ob_int[4]/2)*height]
            labels.append(ob_final)
    f.close()
    labels  = np.array(labels)
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
    names = {n:i for i,n in enumerate(raw)}
    return raw,names

name=  "BISES"
model_name = f"finetune_{name}"
name_result = f"lm_sum_{name}"
topm = 55

path = os.path.join("/home/lionel/theo/active_learning/base-datasets",name)
weights = os.path.join('/home/lionel/theo/active_learning/model/',model_name,'weights/best.pt')
device = "0"
device = select_device(device, batch_size=1)
data = check_dataset(os.path.join(path,"data.yaml"))
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
raw,names = import_lab(os.path.join(path,"data.yaml"))

path_data, img_size = info_model(os.path.join('/home/lionel/theo/active_learning/model/',model_name,"opt.yaml"))
batch_size = 1
task='val'
stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
imgsz = check_img_size(img_size, s=stride)
pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
task = task if task in ('train', 'val', 'test') else 'val'  # 
dataloader = create_dataloader(data[task],
                                    imgsz,
                                    batch_size,
                                    stride,
                                    False,
                                    pad=pad,
                                    rect=rect,
                                    workers=8,
                                    prefix=colorstr(f'{task}: '))[0]

filenames_val = [] 
valid = []
for (im, targets, path, shapes) in tqdm(dataloader) :
    filenames_val.append(os.path.basename(path[0]))

    im = im.to(device, non_blocking=True)
    im = im.half() if False else im.float()  # uint8 to fp16/32
    im /= 255 
    
    p = model(im)
    p = non_max_suppression(p)
    p=  np.array([array[6:-2].cpu() for array in p[0]])
    if len(p) == 0 : 
        valid.append(0)
        continue
    
    sorted_probs = np.sort(p, axis=1)[:, ::-1]

    least_margins = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])

    summ = np.sum(least_margins)
    valid.append(summ)


def top_indices(lst, m):
    print("LST",np.shape(lst))
    indexed_list = [(val, idx) for idx, val in enumerate(lst)]
    sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
    top_indices = [idx for val, idx in sorted_list[:m]]

    top = []
    for i in range(len(lst)):
        top.append( -1 if i in top_indices else  1)

    return top


results = top_indices(np.array(valid),topm)

try:
    os.mkdir(os.path.join("results",name_result))
except:
    pass



text = "Column 1\tColumn 2\n"
for a, b in zip(filenames_val, results):
    text += f"{a}\t{b}\n"
    print(f"{a}\t{b}\n")

with open(os.path.join("results",name_result,"outputval.txt"), 'w') as file:
    file.write(text)


print(name_result)