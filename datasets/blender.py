import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from datasets.colmap import normalize_poses


class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length
        fx = self.focal
        fy = self.focal
        # Expose intrinsics for consumers (e.g. the multi-view mesh losses). cx/cy match the
        # principal point passed to get_ray_directions below (self.w//2, self.h//2).
        self.fx, self.fy = fx, fy
        self.cx, self.cy = self.w // 2, self.h // 2

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            filename = os.path.basename(img_path)
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            mask_method = self.config.get('mask_method', "alpha")

            if mask_method == "file":
                mask_path = os.path.join(self.config.root_dir, f"{self.split}_mask/" + filename)
                mask = Image.open(mask_path).convert('L') # (H, W, 1)
                mask = mask.resize(self.img_wh, Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]
            elif mask_method == "alpha":
                mask = img[..., -1]
            else:
                mask = torch.ones_like(img[...,0], device=img.device)

            self.all_fg_masks.append(mask) # (h, w)
            self.all_images.append(img[...,:3])

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float(), \
            torch.stack(self.all_images, dim=0).float(), \
            torch.stack(self.all_fg_masks, dim=0).float()

        self.all_c2w = torch.cat([self.all_c2w, torch.tensor([[[0,0,0,1]]], dtype=torch.float32, device = self.all_c2w.device).expand(self.all_c2w.shape[0], -1, -1)], dim=1)
        pts3d = torch.empty((0, 3), dtype=torch.float32, device=self.all_c2w.device)
        if self.config.get('normalize_cameras', True):
            all_c2w, pts3d, scale, transform = normalize_poses(self.all_c2w[:,:3,:], pts3d, up_est_method=self.config.up_est_method, center_est_method=self.config.center_est_method)
            all_c2w = torch.cat([all_c2w, torch.tensor([[[0,0,0,1]]], dtype=torch.float32).expand(all_c2w.shape[0], -1, -1)], dim=1)
            self.all_c2w = all_c2w
            self.scale = scale
            self.transform = transform
        else:
            self.scale = 1.0
            self.transform = torch.eye(4, dtype=torch.float32, device=self.all_c2w.device).to(self.rank)

        

        ndc = torch.tensor([
            [2.0 * fx / w, 0, 0, 0],
            [0, -2.0 * fy / h, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), - (2 * self.far * self.near) / (self.far - self.near)],
            #[0, 0, -(camera_parameters['f'] + camera_parameters['n']) / (camera_parameters['f'] - camera_parameters['n']), - (2 * camera_parameters['f'] * camera_parameters['n']) / (camera_parameters['f'] - camera_parameters['n'])],
            [0, 0, -1, 0]
            ], dtype=torch.float32, device = self.all_c2w.device)
        all_w2c = torch.linalg.inv(self.all_c2w) 
        self.all_mvp = ndc @ all_w2c

        self.all_w2c = all_w2c.to(self.rank)
        self.all_c2w = self.all_c2w.to(self.rank)
        self.all_mvp = self.all_mvp.to(self.rank)
        self.pixel_size = (1.0 / fx, 1.0 / fy)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
        

class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            #num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
