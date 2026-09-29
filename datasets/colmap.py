import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
import open3d as o3d


def get_normalization_transform(imdata, pts, up_est_method, center_est_method, scale_est_method):
    all_c2w = []
    for i, d in enumerate(imdata.values()):
        R = d.qvec2rotmat()
        t = d.tvec.reshape(3, 1)
        c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
        c2w[:,1:3] *= -1. # COLMAP => OpenGL
        c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)
        all_c2w.append(c2w)
    all_c2w = torch.stack(all_c2w, dim=0)   
    _, _, scale, transform = normalize_poses(all_c2w[:,:3,:], pts, up_est_method, center_est_method, scale_est_method) 
    return scale, transform


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method, scale_est_method = 'min_cam', scale_mult = 1.0):
    # scale_mult>1 pushes cameras FARTHER from the origin after normalization
    # (effective scale /= scale_mult). Use it to fit a scene whose cameras sit
    # too close-in (small camera radius vs model.radius) into the model's inner
    # region, so the fixed sphere-init radii land inside the camera hull and the
    # tet grid resolves the foreground instead of over-resolving the background.
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        if scale_est_method == 'min_cam':
            scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        elif scale_est_method == 'median_cam':
            scale = poses_norm[...,3].norm(p=2, dim=-1).median()
        elif scale_est_method == 'max_cam':
            scale = poses_norm[...,3].norm(p=2, dim=-1).max()
        else:
            raise NotImplementedError(f'Unknown scale estimation method: {scale_est_method}')
        scale = scale / scale_mult
        poses_norm[...,3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        if scale_est_method == 'min_cam':
            scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        elif scale_est_method == 'median_cam':
            scale = poses_norm[...,3].norm(p=2, dim=-1).median()
        elif scale_est_method == 'max_cam':
            scale = poses_norm[...,3].norm(p=2, dim=-1).max()
        else:
            raise NotImplementedError(f'Unknown scale estimation method: {scale_est_method}')
        scale = scale / scale_mult
        poses_norm[...,3] /= scale

        # apply the transformation to the point cloud
        #pts = (torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1) @ inv_trans.T)[:, :3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale

    return poses_norm, pts, scale, inv_trans

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        if not ColmapDatasetBase.initialized:
            camdata = read_cameras_binary(os.path.join(self.config.root_dir, 'sparse/0/cameras.bin'))

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] * factor
                fy = camdata[1].params[1] * factor
                cx = camdata[1].params[2] * factor
                cy = camdata[1].params[3] * factor
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            
            directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank)
            far = self.config.get('far', 1000.0)
            near = self.config.get('near', 0.01)
            # Original NDC matrix (assumes cx=w/2, cy=h/2)
            # ndc = torch.tensor([
            #         [2.0 * fx / w, 0, 0, 0],
            #         [0, -2.0 * fy / h, 0, 0],
            #         [0, 0, -(far+near)/(far-near), - (2 * far * near) / (far - near)],
            #         #[0, 0, -(camera_parameters['f'] + camera_parameters['n']) / (camera_parameters['f'] - camera_parameters['n']), - (2 * camera_parameters['f'] * camera_parameters['n']) / (camera_parameters['f'] - camera_parameters['n'])],
            #         [0, 0, -1, 0]
            #     ], dtype=torch.float32)

            # Corrected NDC matrix accounting for principal point offset
            ndc = torch.tensor([
                    [2.0 * fx / w, 0,           1.0 - 2.0 * cx / w,       0],
                    [0,           -2.0 * fy / h, 1.0 - 2.0 * cy / h,       0],
                    [0,            0,           -(far+near)/(far-near), - (2 * far * near) / (far - near)],
                    [0,            0,           -1,                       0]
                ], dtype=torch.float32)

            intrinsics = torch.tensor([
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0 ],
                [0, 0, 0, 1 ]
            ], dtype=torch.float32)
            imdata = read_images_binary(os.path.join(self.config.root_dir, 'sparse/0/images.bin'))

            apply_mask = self.config.mask_method in ['file', 'alpha'] 
            if (self.config.mask_method == "file"):
                mask_dir = os.path.join(self.config.root_dir, 'masks')
                if (not os.path.exists(mask_dir)):
                    mask_dir = os.path.join(self.config.root_dir, 'mask')
                if (not os.path.exists(mask_dir)):
                    print(f"Mask directory {mask_dir} does not exist!")
                    apply_mask = False



            all_c2w, all_images, all_fg_masks, all_images_grayscale = [], [], [], []

            for i, d in enumerate(imdata.values()):
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
                c2w[:,1:3] *= -1. # COLMAP => OpenGL
                c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)
                all_c2w.append(c2w)
                if self.split in ['train', 'val']:
                    img_path = os.path.join(self.config.root_dir, 'images', d.name)
                    img = Image.open(img_path)
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)
                    img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                    if self.config.mask_method == "file":
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[1:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    elif self.config.mask_method == "alpha":
                        mask = img[..., 3]
                    else:
                        mask = torch.ones_like(img[...,0], device=img.device)

                    img = img[...,:3]
                    all_fg_masks.append(mask) # (h, w)
                    all_images.append(img)
                    all_images_grayscale.append(0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2])
            
            all_c2w = torch.stack(all_c2w, dim=0)   
            pts3d = read_points3d_binary(os.path.join(self.config.root_dir, 'sparse/0/points3D.bin'))
            pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()
            if self.config.get('normalize_cameras', True):
                all_c2w, pts3d, scale, transform = normalize_poses(all_c2w[:,:3,:], pts3d, up_est_method=self.config.up_est_method, center_est_method=self.config.center_est_method, scale_est_method=self.config.get('scale_est_method', 'min_cam'), scale_mult=self.config.get('scale_mult', 1.0))
                all_c2w = torch.cat([all_c2w, torch.tensor([[[0,0,0,1]]], dtype=torch.float32).expand(all_c2w.shape[0], -1, -1)], dim=1)
            else:
                _, _, scale, transform = normalize_poses(all_c2w[:,:3,:], pts3d, up_est_method=self.config.up_est_method, center_est_method=self.config.center_est_method, scale_est_method=self.config.get('scale_est_method', 'min_cam'), scale_mult=self.config.get('scale_mult', 1.0))

            #mesh = o3d.io.read_triangle_mesh(os.path.join(self.config.root_dir, 'mvs/pc_meshed.ply'))
            
            #transform point cloud
            # mesh_vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
            # mesh_vertices = (torch.cat([mesh_vertices, torch.ones_like(mesh_vertices[:, :1])], dim=-1) @ transform.T)[:, :3]
            # mesh_vertices = mesh_vertices / scale
            # #update vertices
            # mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices.numpy())
            # mesh_normals = torch.from_numpy(np.asarray(mesh.vertex_normals)).float()
            # mesh_normals = (transform[:3,:3] @ mesh_normals.T).T
            # mesh.vertex_normals = o3d.utility.Vector3dVector(mesh_normals.numpy())
            # o3d.io.write_triangle_mesh(os.path.join(self.config.root_dir, 'mvs/pc_meshed_transformed.ply'), mesh)

            # point_cloud = o3d.io.read_point_cloud(os.path.join(self.config.root_dir, 'vggt/fused_voxel_point_cloud_0.005.ply'))
            # pc_positions = torch.from_numpy(np.asarray(point_cloud.points)).float()
            # pc_positions = (torch.cat([pc_positions, torch.ones_like(pc_positions[:, :1])], dim=-1) @ transform.T)[:, :3]
            # pc_positions = pc_positions / scale
            # point_cloud.points = o3d.utility.Vector3dVector(pc_positions.numpy())
            # o3d.io.write_point_cloud(os.path.join(self.config.root_dir, 'vggt/fused_voxel_point_cloud_0.005_transformed.ply'), point_cloud)

            #self.v_pos = (transform @ torch.cat([self.v_pos, torch.ones_like(self.v_pos[:,0:1])], dim=-1)[...,None])[:,:3,0]
            #if self.v_nrm is not None:
            #    self.v_nrm = transform[:3,:3] @ self.v_nrm.T

            
            all_w2c = torch.linalg.inv(all_c2w) 
            all_mvp = ndc @ all_w2c


            # ------------------------------------------------------------
            # Compute nearest camera ids (multi-view neighbors)
            # ------------------------------------------------------------
            with torch.no_grad():
                device = all_c2w.device
                N = all_c2w.shape[0]

                # Camera centers: (N, 3)
                camera_centers = all_c2w[:, :3, 3]

                # Viewing directions in world space: (N, 3)
                # OpenGL convention: camera looks along -Z
                forward_cam = torch.tensor([0.0, 0.0, -1.0], device=device)
                view_dirs = (all_c2w[:, :3, :3] @ forward_cam)
                view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)

                # Pairwise distances: (N, N)
                diss = torch.norm(
                    camera_centers[:, None, :] - camera_centers[None, :, :],
                    dim=-1
                )

                # Pairwise angles (degrees): (N, N)
                dots = torch.sum(view_dirs[:, None, :] * view_dirs[None, :, :], dim=-1)
                dots = torch.clamp(dots, -1.0 + 1e-6, 1.0 - 1e-6)
                angles = torch.acos(dots) * 180.0 / torch.pi

                diss_np = diss.cpu().numpy()
                angles_np = angles.cpu().numpy()

                nearest_cam_ids = []

                for i in range(N):
                    # Sort by distance (primary), angle (secondary)
                    sorted_indices = np.lexsort((angles_np[i], diss_np[i]))

                    # Exclude self explicitly
                    sorted_indices = sorted_indices[sorted_indices != i]

                    # Apply constraints
                    mask = (
                        (angles_np[i][sorted_indices] < self.config.get('multi_view_max_angle', 30)) &
                        (diss_np[i][sorted_indices] > self.config.get('multi_view_min_dis', 0.01)) &
                        (diss_np[i][sorted_indices] < self.config.get('multi_view_max_dis', 1.5))
                    )

                    valid_indices = sorted_indices[mask]

                    # Limit number of neighbors
                    k = min(self.config.get('multi_view_num' ,8), len(valid_indices))
                    nearest_cam_ids.append(valid_indices[:k].tolist())

            ColmapDatasetBase.properties = {
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
                'scale': scale,
                'transform': transform,
                'pixel_size': (1.0 / fx, 1.0 / fy),
                'nearest_cam_ids': nearest_cam_ids
            }

            ColmapDatasetBase.initialized = True
        
        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test' and self.config.get("spherical_test_set", False) > 0:
            all_c2w = create_spheric_poses(self.all_c2w[:,:3,3], n_steps=self.config.n_test_traj_steps)
            all_c2w = torch.cat([all_c2w, torch.tensor([[[0,0,0,1]]], dtype=torch.float32).expand(all_c2w.shape[0], -1, -1)], dim=1)
            self.all_c2w = all_c2w
            self.all_w2c = torch.linalg.inv(self.all_c2w)
            self.all_mvp = self.ndc @ self.all_w2c
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.all_images_grayscale = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
        else:
            self.all_images = torch.stack(self.all_images, dim=0).float()
            self.all_images_grayscale = torch.stack(self.all_images_grayscale, dim=0).float()
            self.all_fg_masks =  torch.stack(self.all_fg_masks, dim=0).float()


        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in self.pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w = self.all_c2w.float().to(self.rank)
        self.all_w2c = self.all_w2c.float().to(self.rank)
        self.all_mvp = self.all_mvp.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
        

class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split, view_subset=None):
        self.setup(config, split)
        # see bmvs.py: serve only these view ids (sparse-train validation)
        self.view_subset = [int(v) for v in view_subset] if view_subset else None

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.config, self.config.get('val_split', 'train'),
                                             view_subset=self.config.get('val_view_ids', None))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')         

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
