"""BlendedMVS dataloader (IDR / NeuS layout).

Layout per scene under `load/bmvs/<scene>/`:
    image/{:03d}.png         RGB images
    mask/{:03d}.png          (optional) foreground masks
    cameras.npz              world_mat_i, scale_mat_i for i in 0..N-1
    gt_pts.ply               (optional) ground-truth point cloud
    normal/                  (optional, unused)

`(world_mat_i @ scale_mat_i)` projects from the IDR-normalized world
(roughly the unit sphere) to image pixels. We decompose that 3x4 matrix
to recover (K, c2w) per image, and convert the pose from OpenCV into
the framework's OpenGL convention (camera looks down -Z) by flipping
the Y/Z columns of c2w.

Mirrors the property surface of `datasets.colmap.ColmapDatasetBase`
so downstream systems treat BMVS identically. Notable differences
from the COLMAP path:
  * no point-cloud-based normalisation: BMVS is already normalised,
    so `scale=1.0` and `transform=I`.
  * `pts3d` is loaded from `gt_pts.ply` if present, otherwise an
    empty (0, 3) tensor placeholder.

Mask handling matches `colmap.py`'s `mask_method` flag:
  * 'file'  -> read from mask/{:03d}.png  (BMVS default)
  * 'alpha' -> use the image's alpha channel (BMVS has none, falls
               back to all-ones with a warning)
  * anything else -> all-ones mask
"""

import os
import math
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from datasets.colmap import normalize_poses
from utils.misc import get_rank


def _load_K_Rt_from_P(P):
    """Decompose a 3x4 projection matrix into (K, c2w_4x4) in OpenCV convention."""
    import cv2  # local import: keeps datasets.bmvs cheap to import when unused
    K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose


def _maybe_load_gt_points(root_dir):
    ply_path = os.path.join(root_dir, 'gt_pts.ply')
    if not os.path.exists(ply_path):
        return torch.zeros((0, 3), dtype=torch.float32)
    try:
        from plyfile import PlyData
        data = PlyData.read(ply_path)
        v = data['vertex']
        pts = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=-1)
        return torch.from_numpy(pts).float()
    except Exception as e:
        print(f"[bmvs] WARN: failed to read {ply_path} ({e}); using empty point cloud.")
        return torch.zeros((0, 3), dtype=torch.float32)


def create_spheric_poses(cameras, n_steps=120):
    """Same convention as colmap.py: a horizontal orbit around z=mean_h at radius r."""
    center = torch.as_tensor([0., 0., 0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d ** 2 - mean_h ** 2).clamp_min(1e-6).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:, None]], dim=1)
        all_c2w.append(c2w)
    return torch.stack(all_c2w, dim=0)


class BMVSDatasetBase():
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        if not BMVSDatasetBase.initialized:
            root = self.config.root_dir
            cam_file = os.path.join(root, self.config.get('cameras_file', 'cameras.npz'))
            assert os.path.exists(cam_file), f"cameras file not found: {cam_file}"
            cams = np.load(cam_file)

            # Discover frames by enumerating world_mat_i (BMVS scenes vary in size,
            # 56-123 frames is typical, named 000.png .. (N-1).png).
            n_images = max(int(k.split('_')[-1]) for k in cams.files if k.startswith('world_mat_')) + 1

            image_dir = os.path.join(root, 'image')
            # Detect filename padding by probing the first image (BMVS uses 3-digit, but
            # be liberal in what we accept).
            sample_idx_str = None
            for w_pad in (3, 4, 5, 6):
                cand = os.path.join(image_dir, f"{0:0{w_pad}d}.png")
                if os.path.exists(cand):
                    sample_idx_str = w_pad
                    break
            assert sample_idx_str is not None, (
                f"Could not find image 000.png (or 0000.png ...) in {image_dir}")
            self._idx_pad = sample_idx_str

            sample_path = os.path.join(image_dir, f"{0:0{self._idx_pad}d}.png")
            with Image.open(sample_path) as im:
                W, H = im.size  # PIL returns (W, H)

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            # Decompose the first projection matrix to get the (constant) intrinsics.
            # BMVS uses one intrinsic per scene; verified by inspection on a handful
            # of scenes. If your scene varies, this is the place to extend.
            P0 = (cams['world_mat_0'] @ cams['scale_mat_0'])[:3, :4]
            K0, _ = _load_K_Rt_from_P(P0.astype(np.float64))
            fx = float(K0[0, 0]) * factor
            fy = float(K0[1, 1]) * factor
            cx = float(K0[0, 2]) * factor
            cy = float(K0[1, 2]) * factor

            directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank)
            far = self.config.get('far', 1000.0)
            near = self.config.get('near', 0.01)
            # Original NDC matrix (assumes cx=w/2, cy=h/2)
            # ndc = torch.tensor([
            #     [2.0 * fx / w, 0, 0, 0],
            #     [0, -2.0 * fy / h, 0, 0],
            #     [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            #     [0, 0, -1, 0],
            # ], dtype=torch.float32)

            # Corrected NDC matrix accounting for principal point offset
            ndc = torch.tensor([
                [2.0 * fx / w, 0,           1.0 - 2.0 * cx / w,       0],
                [0,           -2.0 * fy / h, 1.0 - 2.0 * cy / h,       0],
                [0,            0,           -(far + near) / (far - near), -(2 * far * near) / (far - near)],
                [0,            0,           -1,                       0],
            ], dtype=torch.float32)


            # Mask handling — same flag semantics as datasets/colmap.py.
            mask_method = self.config.get('mask_method', 'file')
            apply_mask = mask_method in ['file', 'alpha']
            mask_dir = None
            if mask_method == 'file':
                cand_dirs = [os.path.join(root, 'mask'), os.path.join(root, 'masks')]
                cand_dirs = [d for d in cand_dirs if os.path.isdir(d)]
                if cand_dirs:
                    mask_dir = cand_dirs[0]
                else:
                    print(f"[bmvs] WARN: mask_method='file' but no mask/ folder under {root}; "
                          f"falling back to all-ones masks.")
                    apply_mask = False
            elif mask_method == 'alpha':
                # BMVS PNGs are RGB without alpha — no actual alpha to read.
                print(f"[bmvs] WARN: mask_method='alpha' is unusual for BMVS (RGB images, "
                      f"no alpha). Set mask_method: file to read mask/{{:0{self._idx_pad}d}}.png.")

            # ---- Monocular normal loading ----
            # Optional per-image normals from a mono predictor (Omnidata, DSINE, ...).
            # Stored under `<root>/normal/{:0Nd}.{npy,png}`. Assumed to be in camera
            # space; convention is configurable (default OpenCV: x-right, y-down,
            # z-forward). They are converted to the framework's camera convention
            # (OpenGL: x-right, y-up, z-back) so the loss code can do a plain dot
            # product with `w2c @ rendered_normal`. Camera-space normals are
            # invariant to world normalization (normalize_poses rotates world AND
            # cameras together), so we don't need to apply `transform` here.
            mono_dir = os.path.join(root, 'normal')
            has_mono_normal = os.path.isdir(mono_dir)
            mono_convention = self.config.get('mono_normal_convention', 'opencv')
            if has_mono_normal:
                # Sign flip applied to convert the predictor's convention to the
                # framework-internal OpenGL camera frame. Use `mono_normal_sign`
                # for full manual control (e.g. [-1,-1,-1] if the mesh normals
                # also point inward).
                manual_sign = self.config.get('mono_normal_sign', None)
                if manual_sign is not None:
                    mono_sign = torch.tensor([float(s) for s in manual_sign])
                elif mono_convention == 'opencv':
                    mono_sign = torch.tensor([1.0, -1.0, -1.0])
                elif mono_convention == 'opengl':
                    mono_sign = torch.tensor([1.0, 1.0, 1.0])
                else:
                    raise ValueError(f"Unknown mono_normal_convention: {mono_convention}")
                print(f"[bmvs] mono normal convention='{mono_convention}', sign={mono_sign.tolist()}")

            all_c2w, all_images, all_fg_masks, all_images_grayscale, all_mono_normals = [], [], [], [], []
            for i in range(n_images):
                P = (cams[f'world_mat_{i}'] @ cams[f'scale_mat_{i}'])[:3, :4]
                _, c2w = _load_K_Rt_from_P(P.astype(np.float64))
                c2w = torch.from_numpy(c2w).float()
                # OpenCV (right, down, forward) -> OpenGL (right, up, back) by flipping y, z columns.
                c2w[:3, 1:3] *= -1.0
                all_c2w.append(c2w)

                if self.split in ['train', 'val']:
                    img_path = os.path.join(image_dir, f"{i:0{self._idx_pad}d}.png")
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)  # (H, W, 3)
                    img = img.to(self.rank) if self.config.get('load_data_on_gpu', False) else img.cpu()

                    if apply_mask and mask_method == 'file' and mask_dir is not None:
                        mp = os.path.join(mask_dir, f"{i:0{self._idx_pad}d}.png")
                        mask = Image.open(mp).convert('L')
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]  # (H, W)
                    elif apply_mask and mask_method == 'alpha':
                        # No alpha in BMVS; degrade gracefully to an all-ones mask.
                        mask = torch.ones_like(img[..., 0], device=img.device)
                    else:
                        mask = torch.ones_like(img[..., 0], device=img.device)

                    all_fg_masks.append(mask)
                    all_images.append(img[..., :3])
                    all_images_grayscale.append(0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])

                    if has_mono_normal:
                        # Try a handful of common naming conventions used by
                        # mono-normal predictors / preprocessing scripts.
                        stem = f"{i:0{self._idx_pad}d}"
                        cand_npy = [os.path.join(mono_dir, f"{stem}.npy"),
                                    os.path.join(mono_dir, f"{stem}_normal.npy")]
                        cand_png = [os.path.join(mono_dir, f"{stem}.png"),
                                    os.path.join(mono_dir, f"{stem}_normal.png")]
                        npy_path = next((p for p in cand_npy if os.path.exists(p)), None)
                        png_path = next((p for p in cand_png if os.path.exists(p)), None)
                        if npy_path is not None:
                            n_arr = np.load(npy_path).astype(np.float32)  # in [-1, 1]
                            # Squeeze a leading batch dim (e.g. (1, 3, H, W)).
                            if n_arr.ndim == 4 and n_arr.shape[0] == 1:
                                n_arr = n_arr[0]
                            # Predictors sometimes save (3, H, W) instead of (H, W, 3).
                            if n_arr.ndim == 3 and n_arr.shape[0] == 3 and n_arr.shape[-1] != 3:
                                n_arr = n_arr.transpose(1, 2, 0)
                            n_t = torch.from_numpy(n_arr)
                            # Resize via bilinear; re-normalization happens below.
                            n_t = F.interpolate(n_t.permute(2, 0, 1)[None], size=(h, w),
                                                mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
                        elif png_path is not None:
                            n_img = Image.open(png_path).convert('RGB').resize(img_wh, Image.BILINEAR)
                            n_t = TF.to_tensor(n_img).permute(1, 2, 0)  # (H, W, 3) in [0, 1]
                            n_t = n_t * 2.0 - 1.0  # to [-1, 1]
                        else:
                            print(f"[bmvs] WARN: missing mono normal for image {stem}; "
                                  f"using zero normals (excluded by loss mask).")
                            n_t = torch.zeros((h, w, 3))
                        # Convert predictor convention -> framework OpenGL camera frame.
                        n_t = n_t * mono_sign
                        # Re-normalize after interpolation / convention flip; keep zero
                        # vectors zero (predictor sometimes outputs 0 outside the FoV).
                        norm = torch.linalg.norm(n_t, dim=-1, keepdim=True).clamp(min=1e-6)
                        n_t = torch.where(norm > 1e-3, n_t / norm, torch.zeros_like(n_t))
                        n_t = n_t.to(self.rank) if self.config.get('load_data_on_gpu', False) else n_t.cpu()
                        all_mono_normals.append(n_t)

            all_c2w = torch.stack(all_c2w, dim=0)

            # Mirror COLMAP normalisation logic
            pts3d = _maybe_load_gt_points(root)
            if self.config.get('normalize_cameras', True):
                all_c2w, pts3d, scale, transform = normalize_poses(
                    all_c2w[:,:3,:], pts3d, 
                    up_est_method=self.config.get('up_est_method', 'camera'), 
                    center_est_method=self.config.get('center_est_method', 'lookat'), 
                    scale_est_method=self.config.get('scale_est_method', 'min_cam')
                )
                all_c2w = torch.cat([all_c2w, torch.tensor([[[0,0,0,1]]], dtype=torch.float32).expand(all_c2w.shape[0], -1, -1)], dim=1)
            else:
                #set scale and transform to identiy
                scale = 1.0
                transform = torch.eye(4)

            all_w2c = torch.linalg.inv(all_c2w)
            all_mvp = ndc @ all_w2c

            # ---- Nearest-camera neighbours (verbatim from colmap.py) ----
            with torch.no_grad():
                device = all_c2w.device
                N = all_c2w.shape[0]
                camera_centers = all_c2w[:, :3, 3]
                forward_cam = torch.tensor([0.0, 0.0, -1.0], device=device)
                view_dirs = (all_c2w[:, :3, :3] @ forward_cam)
                view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)

                diss = torch.norm(camera_centers[:, None, :] - camera_centers[None, :, :], dim=-1)
                dots = torch.sum(view_dirs[:, None, :] * view_dirs[None, :, :], dim=-1)
                dots = torch.clamp(dots, -1.0 + 1e-6, 1.0 - 1e-6)
                angles = torch.acos(dots) * 180.0 / torch.pi

                diss_np = diss.cpu().numpy()
                angles_np = angles.cpu().numpy()
                nearest_cam_ids = []
                for i in range(N):
                    sorted_indices = np.lexsort((angles_np[i], diss_np[i]))
                    sorted_indices = sorted_indices[sorted_indices != i]
                    keep = (
                        (angles_np[i][sorted_indices] < self.config.get('multi_view_max_angle', 30)) &
                        (diss_np[i][sorted_indices] > self.config.get('multi_view_min_dis', 0.01)) &
                        (diss_np[i][sorted_indices] < self.config.get('multi_view_max_dis', 1.5))
                    )
                    valid_indices = sorted_indices[keep]
                    k = min(self.config.get('multi_view_num', 8), len(valid_indices))
                    nearest_cam_ids.append(valid_indices[:k].tolist())

            BMVSDatasetBase.properties = {
                'ndc': ndc,
                'w': w,
                'h': h,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'img_wh': img_wh,
                'factor': factor,
                'has_mask': apply_mask,
                'apply_mask': apply_mask,
                'directions': directions,
                'pts3d': pts3d,
                'all_c2w': all_c2w,
                'all_w2c': all_w2c,
                'all_mvp': all_mvp,
                'all_images': all_images,
                'all_images_grayscale': all_images_grayscale,
                'all_fg_masks': all_fg_masks,
                'all_mono_normals': all_mono_normals,
                'has_mono_normal': has_mono_normal,
                'scale': scale,
                'transform': transform,
                'pixel_size': (1.0 / fx, 1.0 / fy),
                'nearest_cam_ids': nearest_cam_ids,
            }
            BMVSDatasetBase.initialized = True

        for k, v in BMVSDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test' and self.config.get('spherical_test_set', False):
            all_c2w = create_spheric_poses(self.all_c2w[:, :3, 3],
                                           n_steps=self.config.n_test_traj_steps)
            all_c2w = torch.cat([all_c2w,
                                 torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32).expand(all_c2w.shape[0], -1, -1)],
                                dim=1)
            self.all_c2w = all_c2w
            self.all_w2c = torch.linalg.inv(self.all_c2w)
            self.all_mvp = self.ndc @ self.all_w2c
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.all_images_grayscale = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.all_mono_normals = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
        else:
            self.all_images = torch.stack(self.all_images, dim=0).float()
            self.all_images_grayscale = torch.stack(self.all_images_grayscale, dim=0).float()
            self.all_fg_masks = torch.stack(self.all_fg_masks, dim=0).float()
            if self.has_mono_normal and len(self.all_mono_normals) > 0:
                self.all_mono_normals = torch.stack(self.all_mono_normals, dim=0).float()
            else:
                self.all_mono_normals = torch.zeros((len(self.all_images), self.h, self.w, 3), dtype=torch.float32)

        self.all_c2w = self.all_c2w.float().to(self.rank)
        self.all_w2c = self.all_w2c.float().to(self.rank)
        self.all_mvp = self.all_mvp.float().to(self.rank)
        if self.config.get('load_data_on_gpu', False):
            self.all_images = self.all_images.to(self.rank)
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
            self.all_mono_normals = self.all_mono_normals.to(self.rank)


class BMVSDataset(Dataset, BMVSDatasetBase):
    def __init__(self, config, split, view_subset=None):
        self.setup(config, split)
        # Restrict served indices to an explicit view list (e.g. the sparse
        # TRAIN views for validation) -- otherwise limit_val_batches=N always
        # renders views 0..N-1, which are NOT the training views.
        self.view_subset = [int(v) for v in view_subset] if view_subset else None

    def __len__(self):
        return len(self.view_subset) if self.view_subset is not None else len(self.all_images)

    def __getitem__(self, index):
        if self.view_subset is not None:
            return {'index': self.view_subset[index]}
        return {'index': index}


class BMVSIterableDataset(IterableDataset, BMVSDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('bmvs')
class BMVSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BMVSIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BMVSDataset(self.config, self.config.get('val_split', 'train'),
                                           view_subset=self.config.get('val_view_ids', None))
        if stage in [None, 'test']:
            self.test_dataset = BMVSDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = BMVSDataset(self.config, 'train')

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=None,
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
