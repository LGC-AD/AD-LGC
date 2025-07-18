import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
# import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color
import torch.nn.functional as F

def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps, method, root_out, dataset_name):
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)
    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        img_path = f'{out_dir}/{img_num}_img.png'
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        img_rec = img * std[:, None, None] + mean[:, None, None]
        # RGB image
        img_rec = Image.fromarray((img_rec * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        img_rec.save(img_path)
        # RGB image with anomaly map
        anomaly_map = anomaly_map / anomaly_map.max()
        anomaly_map = cm.jet(anomaly_map)
        # anomaly_map = cm.rainbow(anomaly_map)
        anomaly_map = (anomaly_map[:, :, :3] * 255).astype('uint8')
        anomaly_map = Image.fromarray(anomaly_map)
        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)
        # mask
        img_mask = Image.fromarray((img_mask * 255).astype(np.uint8).transpose(1, 2, 0).repeat(3, axis=2))
        img_mask.save(mask_path)


def save_vis_rgb(save_path, img_path, imgs, save_name):
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
    for i in range(len(img_path)):
        img_name = os.path.splitext(os.path.basename(img_path[i]))[0]
        img_save_path = os.path.join(save_path, f'{img_name}_{save_name}.png')
        img = imgs[i] * std[:, None, None] + mean[:, None, None]
        img = (img * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(img_save_path)


def save_data(save_path, cls_name, img_path, imgs_mask, anomaly_map, anomaly):
    for i in range(len(img_path)):
        img_name = os.path.splitext(os.path.basename(img_path[i]))[0]
        defect_name = os.path.basename(os.path.dirname(img_path[i]))

        cls = cls_name[i]
        class_path = os.path.join(save_path, cls)
        mask_path = os.path.join(class_path, 'masks')
        map_path = os.path.join(class_path, 'maps')
        anomaly_path = os.path.join(class_path, 'anomalies')
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        if not os.path.exists(map_path):
            os.makedirs(map_path)
        if not os.path.exists(anomaly_path):
            os.makedirs(anomaly_path)

        np.save(os.path.join(mask_path, f'{defect_name}_{img_name}.npy'), imgs_mask[i])
        np.save(os.path.join(map_path, f'{defect_name}_{img_name}.npy'), anomaly_map[i])
        np.save(os.path.join(anomaly_path, f'{defect_name}_{img_name}.npy'), anomaly[i])

def read_data(save_path, cls_name):
    class_path = os.path.join(save_path, cls_name)
    mask_path = os.path.join(class_path, 'masks')
    map_path = os.path.join(class_path, 'maps')
    anomaly_path = os.path.join(class_path, 'anomalies')
    cls_names = []
    imgs_mask = []
    anomaly_map = []
    anomaly = []
    for file in os.listdir(mask_path):
        if file.endswith('.npy'):
            cls_names.append(cls_name)
            imgs_mask.append(np.load(os.path.join(mask_path, file)))
            anomaly_map.append(np.load(os.path.join(map_path, file)))
            anomaly.append(np.load(os.path.join(anomaly_path, file)))
    return imgs_mask, anomaly_map, anomaly, cls_names