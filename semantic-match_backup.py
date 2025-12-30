import os
from functools import lru_cache

import torch
import torch.nn as nn
from einops import rearrange

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches
from PIL import Image
from load_and_save import load_image, save_image, load_image_info, save_image_info
from semantic_mask import get_mask
import time 
import cv2
import numpy as np


base_mask_path = 'mask_folder'
os.makedirs(base_mask_path, exist_ok=True)
_CMAP = cm.get_cmap('turbo')


_SRC_DOMAIN = "woman"
_SRC_IMG_PATH = "flux_batch_0.png" 

_PATCH_ROW = 16
_PATCH_COL = 30

QUANTILE_SHOWN = 1

    

def get_unique_filepath(stream_type, base_path, filename_template, step_idx):

    step_str = f"{step_idx:03d}"
    stream_str = f"{stream_type}"
    name_base = filename_template.format(stream_type=stream_str, step_idx=step_str)
    name_base = name_base.replace(
            f"_step{step_str}", f"step{step_str}_0")
    filepath = os.path.join(base_path, name_base)
    counter = 0
    while os.path.exists(filepath):
        new_name = filename_template.format(stream_type=stream_str, step_idx=step_str) 
        new_name = new_name.replace(
            f"_step{step_str}", f"step{step_str}_{counter}"
        )
        filepath = os.path.join(base_path, new_name)
        counter += 1
    return filepath


def _cosine_sim(src_features: torch.Tensor, tar_features: torch.Tensor) -> torch.Tensor:
    s = nn.functional.normalize(src_features, dim=-1)
    t = nn.functional.normalize(tar_features, dim=-1)
    return s @ t.transpose(1, 2)  # (B,M,N)


def get_top_res(heatmap_ex):
    heatmap_ex = torch.from_numpy(heatmap_ex)
    threshold = torch.quantile(heatmap_ex, 1 - QUANTILE_SHOWN)
    mask = heatmap_ex > threshold
    heatmap_ex = torch.where(mask, heatmap_ex, torch.full_like(heatmap_ex, float('nan')))
    return heatmap_ex.detach().cpu().numpy()
    
    
def get_match(stream_type, src_features: torch.Tensor, tar_features: torch.Tensor, h_src: int = 64, w_src: int = 64, h_tar: int = 64, w_tar: int = 64):
    # рисуем птч
    curtime = time.time()
    src_img = load_image(_SRC_IMG_PATH) 
    W, H = src_img.size
    cell_w = W / w_src
    cell_h = H / h_src
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(src_img)
    rect_src = patches.Rectangle((_PATCH_COL * cell_w, _PATCH_ROW * cell_h),
                                 cell_w, cell_h,
                                 linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect_src)
    ax.set_title("Source patch")
    ax.axis('off')
    patch_path = ("source_patch.png")
    plt.savefig(patch_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    
    # матчим
    if len([f for f in os.listdir("mask_folder") if os.path.isfile(os.path.join("mask_folder", f))]) == 562:
        sim_src2tar_used = _cosine_sim(src_features, tar_features)
        patch_id = _PATCH_ROW * w_src + _PATCH_COL
        sim_row = sim_src2tar_used[0, patch_id, :].to(torch.float32)
        heatmap = sim_row.view(h_tar, w_tar).detach().cpu().numpy()
        heatmap = get_top_res(heatmap)

    else: 
        sim_src2tar_used = torch.ones((1, h_src * w_src, h_tar * w_tar),device=src_features.device,dtype=src_features.dtype,)
        heatmap = np.ones((h_tar, w_tar))
        
    
    step = 1   # заглушка для step

    
    if not os.path.exists("mask_nd"):
        mask = get_mask(_SRC_IMG_PATH, _SRC_DOMAIN)
        mask = cv2.resize(mask, (h_tar, w_tar), interpolation=cv2.INTER_NEAREST)
        mask.astype(np.uint8).tofile("mask_nd")
    else:
        mask = np.fromfile("mask_nd", dtype=np.uint8).reshape((h_tar, w_tar))
    #от сюда
    if (len([f for f in os.listdir("mask_folder") if os.path.isfile(os.path.join("mask_folder", f))]) == 562):
        heatmap = np.where(mask > 0, heatmap, np.nan) #значения внутри объекта, вне наны
        heatmap2 = -np.ones((h_tar, w_tar), dtype=np.int32) #мапа где на кажд патче в маске 0 тупль коорд ближайшего патча в 1 
        
        for x in range(h_tar):
            for y in range(w_tar):
                if np.isfinite(heatmap[x, y]) and heatmap[x, y] > 0:
                    patch_id = x * w_tar + y 
                    heatmap_tmp = sim_src2tar_used[0, patch_id, :].to(torch.float32)
                    best_match = int(torch.argmax(heatmap_tmp))
                    heatmap2[x, y] = best_match
        
        heatmap2.tofile("heatmap_nd")
        template_heatmap = "{stream_type}_heatmap_step{step_idx}.png"
        heatmap_path = get_unique_filepath(stream_type, base_mask_path, template_heatmap, step)
        plt.imsave(heatmap_path, heatmap, cmap=_CMAP, vmin=-1.0, vmax=1.0)

        #до сюда
    
    template_heatmap = "{stream_type}_heatmap_step{step_idx}.png"
    heatmap_path = get_unique_filepath(stream_type, base_mask_path, template_heatmap, step)
    
    if heatmap_path:
        plt.imsave(heatmap_path, heatmap, cmap=_CMAP, vmin=-1.0, vmax=1.0)
        print(f"Сохранено: {heatmap_path}, итерация заняла {time.time() - curtime} секунд")