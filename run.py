import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import base64
import time
import math
import datetime
import os
import seaborn as sns
import scipy 
import multiprocessing as mp 

from joblib import Parallel , delayed
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm.notebook import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from IPython.display import Image, clear_output

import torch
import torchvision
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import defaultdict, deque

# json file 을 읽은후 yolo format으로 bbox를 만들고 image를 decoding하여 image를 해당폴더에 생성합니다.

def xyxy2coco(xyxy): # 코코형식으로 변환
    x1,y1,x2,y2 =xyxy
    w,h =  x2-x1, y2-y1
    return [x1,y1,w,h] 

def xyxy2yolo(xyxy): # 욜로형식으로 변환
    
    x1,y1,x2,y2 =xyxy
    w,h =  x2-x1, y2-y1
    xc = x1 + int(np.round(w/2)) # xmin + width/2
    yc = y1 + int(np.round(h/2)) # ymin + height/2
    return [xc/IMG_SIZE,yc/IMG_SIZE,w/IMG_SIZE,h/IMG_SIZE] 

def scale_bbox(img, xyxy):
    # Get scaling factor
    scale_x = IMG_SIZE/img.shape[1]
    scale_y = IMG_SIZE/img.shape[0]
    
    x1,y1,x2,y2 =xyxy
    x1 = int(np.round(x1*scale_x,4))
    y1 = int(np.round(y1*scale_y, 4))
    x2 = int(np.round(x2*scale_x, 4))
    y2= int(np.round(y2*scale_y, 4))

    return [x1, y1, x2, y2] # xmin, ymin, xmax, ymax

def save_image_label(json_file,mode): 
    with open(json_file,'r') as f: 
        json_file =json.load(f)

    image_id = json_file['file_name'].replace('.json','')
    
    # decode image data
    image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(str(new_image_path / (image_id + '.png')) ,image)
    
    # extract bbox
    origin_bbox = []
    if mode == 'train':
        with open(new_label_path / (image_id + '.txt'), 'w') as f:
            for i in json_file['shapes']: 
                bbox = i['points'][0] + i['points'][2]
                origin_bbox.append(bbox)
                bbox = scale_bbox(image,bbox)
                bbox = xyxy2yolo(bbox)
                
                labels = [categories[i['label']]]+bbox
                f.writelines([f'{i} ' for i in labels] + ['\n']) 
    return origin_bbox

base_path = Path('/content/')
train_files = sorted(glob('/content/train/*'))
test_files = sorted(glob('/content/test/*'))

if len(train_files) == 62622:
  print('데이터를 정상적으로 불러왔습니다.')
else:
  print('데이터 길이에 문제가 있습니다.')

if len(test_files) == 20874:
  print('데이터를 정상적으로 불러왔습니다.')
else:
  print('데이터 길이에 문제가 있습니다.')

IMG_SIZE = 256
train_path = list((base_path /'train').glob('train*'))
test_path = list((base_path / 'test').glob('test*'))

label_info = pd.read_csv((base_path /'class_id_info.csv')) # csv 라벨링 값을 데이터 프레임 형태로 가져옴
categories = {i[0]:i[1]-1 for i in label_info.to_numpy()}
label_info

save_path = Path('./train_data')
new_image_path = save_path / 'images' # image폴더 
new_label_path = save_path / 'labels' # label폴더

new_image_path.mkdir(parents=True,exist_ok=True)
new_label_path.mkdir(parents=True,exist_ok=True)

# data를 생성하기 위해 multiprocessing 적용
tmp = Parallel(n_jobs=mp.cpu_count(),prefer="threads")(delayed(save_image_label)(str(train_json),'train') for train_json in tqdm(train_path))

n = 10
filename = train_path[n].name.replace('.json','.png')
sample = cv2.imread(f'./train_data/images/{filename}')[:,:,::-1].astype(np.uint8)
for i in tmp[n]: 
    i = list(map(int,i))
    sample = cv2.rectangle(sample,(i[0],i[1]),(i[2],i[3]),(0,0,255),1)
plt.imshow(sample)

images_path = list(new_image_path.glob('*'))

train_path_list,valid_path_list = train_test_split(images_path,test_size=0.2,random_state=1)

with open('train_dataset.txt', 'w') as f:
    f.writelines([f'{i}\n' for i in train_path_list])
with open('valid_dataset.txt', 'w') as f:
    f.writelines([f'{i}\n' for i in valid_path_list]) 

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
