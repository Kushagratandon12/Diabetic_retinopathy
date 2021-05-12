import os
import pandas as pd
import glob
import cv2
import time
import numpy as np
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# downloading datasets for COVID-19 data
# api.dataset_download_files('amanneo/diabetic-retinopathy-resized-arranged')
data = "F:\\Project's Kushagra\\Diabetic_retinopathy\\data"

data_all_imgs = glob.glob(data+'\\*\\*.jpeg')
IMG_SIZE = (512, 512)


def pre_process_img(filename, sigmaX=30):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(
        img, (0, 0), sigmaX), -4, 128)
    return img


for img_name in tqdm(data_all_imgs, position=0, leave=True):
    process_img = pre_process_img(img_name)
    save_path = img_name.replace('data', 'processed_data')
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    cv2.imwrite(str(save_path), process_img)
    time.sleep(0.1)