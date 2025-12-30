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
    
    
def get_match(
    global_idx,
    stream_type,
    src_features: torch.Tensor,
    tar_features: torch.Tensor,
    h_src: int = 64,
    w_src: int = 64,
    h_tar: int = 64,
    w_tar: int = 64,
):
    t0 = time.time()

    if not os.path.exists("mask_nd"):
        try:
            mask = get_mask(_SRC_IMG_PATH, _SRC_DOMAIN)
        except Exception as e:
            print(f"ERR% {e}")
            mask = np.ones((h_tar, w_tar))
        mask = cv2.resize(mask, (h_tar, w_tar), interpolation=cv2.INTER_NEAREST)
        mask.astype(np.uint8).tofile("mask_nd")
    else:
        mask = np.fromfile("mask_nd", dtype=np.uint8).reshape(h_tar, w_tar)

    sim = _cosine_sim(src_features, tar_features)  # (1, 4096, 4096)
    patch_id = _PATCH_ROW * w_src + _PATCH_COL
    sim_row = sim[0, patch_id]                     # (4096,)
    heatmap = sim_row.float().view(h_tar, w_tar).cpu().numpy()
    heatmap = get_top_res(heatmap)

    heatmap = np.where(mask > 0, heatmap, np.nan)
    
    best_val, best_idx = torch.max(sim[0], dim=1)
    best_val = best_val.float().detach().cpu().numpy()
    best_idx = best_idx.detach().cpu().numpy()

    mask_flat = (mask.reshape(-1) > 0)

    if (global_idx > 230 and global_idx < 255): #диапазон с ок хитмапами 
        old_val = np.full(h_tar * w_tar, -np.inf, dtype=np.float32)
        old_idx = np.full(h_tar * w_tar, -1, dtype=np.int32)

        if os.path.exists("heatmap_bval_nd"):
            old_val = np.fromfile("heatmap_bval_nd", dtype=np.float32).reshape(-1)
        if os.path.exists("heatmap_nd"):
            old_idx = np.fromfile("heatmap_nd", dtype=np.int32).reshape(-1)
        update_mask = (best_val > old_val) & mask_flat

        old_val[update_mask] =  best_val[update_mask]
        old_idx[update_mask] = best_idx[update_mask]

        old_val.reshape(h_tar, w_tar).tofile("heatmap_bval_nd")
        old_idx.reshape(h_tar, w_tar).tofile("heatmap_nd")    

    path = get_unique_filepath(
        stream_type,
        base_mask_path,
        "{stream_type}_heatmap_step{step_idx}.png",
        step_idx=1,
    )
    plt.imsave(path, heatmap, cmap=_CMAP, vmin=-1.0, vmax=1.0)

    print(f"[MATCH] выполнен, {time.time() - t0:.3f} сек")
