from distutils import util
import os
from matplotlib.pyplot import grid
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss


from models.utils import chunk_batch
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy, SSIM
import render.renderutils as ru
from .loss_utils import ssim, get_img_grad_weight, lncc
from .graphics_utils import patch_warp

@systems.register('radtets-system')
class RadtetsSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def on_train_start(self):
        dataset = self.trainer.datamodule.train_dataloader().dataset
        all_mvp = dataset.all_mvp
        self.model.all_mvp = all_mvp
        self.model.all_campos = dataset.all_c2w[..., :3, 3]
        W, H = dataset.img_wh
        self.model.image_resolution = (H, W)
        #self.prune_invisible_points()
        if self.config.model.get('debug_radial_sdf', False):
            import os
            self.model.dump_radial_sdf(os.path.join(self.save_dir, 'radial_sdf_init.csv'))
        # dump the INIT grid (before any training) so the point distribution /
        # seedspawn / double-sphere init can be inspected directly
        self.model.save_grid(self.get_save_path("grid_init.ply"), scale=dataset.scale,
                             transform=torch.linalg.inv(dataset.transform))

    def on_train_end(self):
        self.model.save_grid(self.get_save_path(f"final_grid.ply"), scale = self.dataset.scale, transform = torch.linalg.inv(self.dataset.transform))
        self.model.save_mesh(self.get_save_path(f"final_mesh.ply"), scale = self.dataset.scale, transform = torch.linalg.inv(self.dataset.transform))


    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays
        self.train_num_images = self.config.model.train_num_images
        self.shuffle_images = self.config.model.get('shuffle_images', True)
        self.current_image_idx = 0
        self.validation_ray_chunk = self.config.system.get('validation_ray_chunk', 2048)

    @torch.no_grad()
    def prune_invisible_points(self):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        mvps = self.dataset.all_mvp
        points = self.model.tetrahedral_grid.primal_points
        v_pos_clip = ru.xfm_points(points[None, ...].clone(), mvps[:,:,:]) # (N, V, 4)
        v_pos_clip = v_pos_clip[..., :3] / v_pos_clip[..., 3:4]
        valid_points_mask = v_pos_clip.abs().max(dim=-1).values <= 1.0
        valid_points_mask = valid_points_mask.any(dim=0) # (V,)
        self.model.tetrahedral_grid.prune_points(~valid_points_mask)
        self.model.save_grid(self.get_save_path(f"pruned-grid.ply"))


	# def on_after_backward(loss):
    #     pass

    def generate_background(self, shape, color = 'white'):
        if color == 'checker':
            background = torch.tensor(util.checkerboard(shape[0:2], 8), dtype=torch.float32, device=self.rank)[None, ...]
        elif color == 'black':
            background = torch.zeros((shape + (3,)), dtype=torch.float32, device=self.rank)
        elif color == 'white':
            background = torch.ones((shape + (3,)), dtype=torch.float32, device=self.rank)
        elif color == 'random':
            background = torch.rand((shape + (3,)), dtype=torch.float32, device=self.rank)
        else:
            raise NotImplementedError
        return background
    
    def _sample_view_indices(self, size):
        """Camera-view sampler for TRAINING. If model.train_view_ids is set
        (sparse-view experiments), sample only from those original view indices;
        otherwise sample uniformly over all views. All cameras stay loaded, so
        all_w2c/all_mvp remain aligned to original view indices."""
        dev = self.dataset.all_images.device
        vids = self.config.model.get('train_view_ids', None)
        if not vids:
            return torch.randint(0, len(self.dataset.all_images), size=size, device=dev)
        if getattr(self, '_train_view_ids_t', None) is None:
            self._train_view_ids_t = torch.tensor(list(vids), dtype=torch.long, device=dev)
        return self._train_view_ids_t[torch.randint(0, len(self._train_view_ids_t), size=size, device=dev)]

    def get_ray_batch(self, num_rays, batch_image_sampling = True, bg_color = 'white'):
        if batch_image_sampling:
            index = self._sample_view_indices((num_rays,))
        else:
            index = self._sample_view_indices((1,))
        c2w = self.dataset.all_c2w[index]
        x = torch.randint(0, self.dataset.w, size=(num_rays,), device=self.dataset.all_images.device)
        y = torch.randint(0, self.dataset.h, size=(num_rays,), device=self.dataset.all_images.device)
        if self.dataset.directions.ndim == 3: # (H, W, 3)
            directions = self.dataset.directions[y, x]
        elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
            directions = self.dataset.directions[index, y, x]
        rays_o, rays_d = get_rays(directions, c2w)
        rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
        fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
        background = self.generate_background((self.train_num_rays,), color = bg_color).to(self.rank)
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + background * (1 - fg_mask[...,None])

        out_batch = {
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'background': background
        }
        # Per-ray mono normal target (camera frame) + per-ray w2c rotation,
        # used by the volumetric mono-normal loss. Pulled here so the loss
        # site can stay simple (no cross-referencing between rays and images).
        if getattr(self.dataset, 'has_mono_normal', False):
            out_batch['mono_normal'] = self.dataset.all_mono_normals[index, y, x].to(self.rank)  # (R, 3)
            out_batch['w2c'] = self.dataset.all_w2c[index]                                       # (R, 4, 4)
        return out_batch

    def get_image_batch(self, index, include_rays = False, bg_color = 'white'):
        c2w = self.dataset.all_c2w[index]
        w2c = self.dataset.all_w2c[index]
        mvp = self.dataset.all_mvp[index]
        #grayscale = self.dataset.all_images_grayscale[index].to(self.rank)
        #nearest_cam_ids = nearest_cam_ids = [self.dataset.nearest_cam_ids[i] for i in index.tolist()]
        campos = c2w[..., :3, 3]
        rgb = self.dataset.all_images[index].to(self.rank)
        fg_mask = self.dataset.all_fg_masks[index].to(self.rank)
        background = self.generate_background(rgb.shape[:-1], color = bg_color).to(self.rank)
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + background * (1 - fg_mask[...,None])
        batch = {
            'c2w': c2w,
            'w2c': w2c,
            'campos': campos,
            'mvp': mvp,
            'rgb': rgb,
            #'grayscale': grayscale,
            'fg_mask': fg_mask,
            'background': background,
            'image_idx': index,
            #'nearest_cam_ids': nearest_cam_ids
        }
        if getattr(self.dataset, 'has_mono_normal', False):
            batch['mono_normal'] = self.dataset.all_mono_normals[index].to(self.rank)

        if include_rays:
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w, jitter = False)
            rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
            batch['rays'] = rays
        
        return batch

    def preprocess_data(self, batch, stage):

        if stage in ['train']:
            if self.C(self.config.system.lambda_volumetric) != 0:
                vol_batch = self.get_ray_batch(self.train_num_rays, batch_image_sampling = self.config.model.batch_image_sampling, bg_color = self.config.system.train_background_color)
                batch['vol_batch'] = vol_batch
            if self.C(self.config.system.lambda_mesh) != 0:
                if self.shuffle_images:
                    mesh_index = self._sample_view_indices((self.train_num_images,))
                else:
                    pool = getattr(self, '_train_view_ids_t', None)
                    if pool is None and self.config.model.get('train_view_ids', None):
                        pool = torch.tensor(list(self.config.model.train_view_ids), dtype=torch.long, device=self.dataset.all_images.device)
                        self._train_view_ids_t = pool
                    n = len(pool) if pool is not None else len(self.dataset.all_images)
                    sel = torch.arange(self.current_image_idx, self.current_image_idx + self.train_num_images, device=self.dataset.all_images.device) % n
                    mesh_index = pool[sel] if pool is not None else sel
                    self.current_image_idx = (self.current_image_idx + self.train_num_images) % n
                mesh_batch = self.get_image_batch(mesh_index, include_rays = True, bg_color = self.config.system.train_background_color)
                batch['mesh_batch'] = mesh_batch
        else:
            index = batch['index']
            # if torch.is_tensor(index):
            #     index = index.unsqueeze(0)   # [1] instead of []
            # else:
            #     index = torch.tensor([index], device=self.dataset.all_images.device)
            val_batch = self.get_image_batch(index, include_rays = True, bg_color = self.config.system.val_background_color)
            batch.update(val_batch)
   
    
    def _build_mesh_f_pix(self, out, gt_rgb, batch):
        """Per-(sample, pixel) total mesh loss with all enabled lambdas baked in.

        Used as `f` in the score-function topology gradient: REINFORCE expects
        the actual scalar objective whose expectation under the topology
        distribution we are minimizing. Returns (N, B, H, W).

        Pixel-decomposable terms are included (RGB MSE/L1/grad-L1/L1-adaptive,
        mask BCE/MSE, normal smoothness). Patch- or sample-level terms (SSIM,
        eikonal, multiview) have no clean per-pixel form and are omitted; this
        introduces a small bias but the dominant photometric+mask signal is
        preserved.
        """
        rgb_pred = out['rgb'][..., 0:3]
        device = rgb_pred.device
        N, B, H, W = rgb_pred.shape[:4]
        f_pix = torch.zeros((N, B, H, W), device=device)
        ml = self.config.system.mesh_loss
        vl = self.config.system.volumetric_loss

        def _add(per_pix, lam):
            if lam == 0 or per_pix is None:
                return
            pad_h = H - per_pix.shape[-2]
            pad_w = W - per_pix.shape[-1]
            if pad_h or pad_w:
                per_pix = F.pad(per_pix, (0, pad_w, 0, pad_h), value=0.0)
            return lam * per_pix

        # RGB MSE / L1
        diff = rgb_pred - gt_rgb
        per_mse = diff.pow(2).mean(-1)
        per_l1 = diff.abs().mean(-1)
        v = _add(per_mse, self.C(ml.lambda_rgb_mse))
        if v is not None: f_pix = f_pix + v
        v = _add(per_l1, self.C(ml.lambda_rgb_l1))
        if v is not None: f_pix = f_pix + v

        # RGB grad L1 (image-gradient L1, half along x + half along y)
        lam_grad = self.C(ml.get('lambda_rgb_grad_l1', 0))
        if lam_grad > 0:
            gx = (rgb_pred[..., :, 1:, :] - rgb_pred[..., :, :-1, :])
            gy = (rgb_pred[..., 1:, :, :] - rgb_pred[..., :-1, :, :])
            gtx = (gt_rgb[..., :, 1:, :] - gt_rgb[..., :, :-1, :])
            gty = (gt_rgb[..., 1:, :, :] - gt_rgb[..., :-1, :, :])
            per_x = (gx - gtx).abs().mean(-1)
            per_y = (gy - gty).abs().mean(-1)
            v = _add(per_x, 0.5 * lam_grad)
            if v is not None: f_pix = f_pix + v
            v = _add(per_y, 0.5 * lam_grad)
            if v is not None: f_pix = f_pix + v

        # RGB L1 adaptive (the config keys it under volumetric_loss but it is
        # also applied to the mesh path; mirror that here.)
        lam_ada = self.C(vl.get("lambda_rgb_l1_adaptive", 0.0))
        if lam_ada > 0:
            specular_w = (1.0 - gt_rgb.mean(dim=-1, keepdim=True) ** 5).clamp(min=0.01)
            per_ada = (diff.abs() * specular_w).mean(-1)
            v = _add(per_ada, lam_ada)
            if v is not None: f_pix = f_pix + v

        # Normal smoothness (image-gradient-weighted)
        lam_ns = self.C(ml.get('lambda_normal_smoothness', 0))
        if lam_ns > 0 and 'geometric_normal' in out:
            normal = out['geometric_normal']
            x_grad_img = (gt_rgb[..., :, 1:, :] - gt_rgb[..., :, :-1, :]).abs().mean(dim=-1, keepdim=True)
            y_grad_img = (gt_rgb[..., 1:, :, :] - gt_rgb[..., :-1, :, :]).abs().mean(dim=-1, keepdim=True)
            x_grad_img = (x_grad_img - x_grad_img.min()) / (x_grad_img.max() - x_grad_img.min() + 1e-7)
            y_grad_img = (y_grad_img - y_grad_img.min()) / (y_grad_img.max() - y_grad_img.min() + 1e-7)
            x_w = (1.0 - x_grad_img).clamp(0., 1.) ** 2
            y_w = (1.0 - y_grad_img).clamp(0., 1.) ** 2
            nx = (normal[..., :, 1:, :] - normal[..., :, :-1, :]).abs().mean(dim=-1, keepdim=True)
            ny = (normal[..., 1:, :, :] - normal[..., :-1, :, :]).abs().mean(dim=-1, keepdim=True)
            per_nx = (x_w * nx).squeeze(-1)
            per_ny = (y_w * ny).squeeze(-1)
            v = _add(per_nx, lam_ns)
            if v is not None: f_pix = f_pix + v
            v = _add(per_ny, lam_ns)
            if v is not None: f_pix = f_pix + v

        # Mask losses (only meaningful when has_mask).
        if 'opacity' in out and self.dataset.has_mask:
            opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1. - 1.e-3)
            gt_mask = batch['fg_mask'].float().unsqueeze(0).expand_as(opacity)
            lam_bce = self.C(ml.lambda_mask_bce)
            if lam_bce > 0:
                per_bce = F.binary_cross_entropy(opacity, gt_mask, reduction='none')
                v = _add(per_bce, lam_bce)
                if v is not None: f_pix = f_pix + v
            lam_mse = self.C(ml.lambda_mask_mse)
            if lam_mse > 0:
                per_msk = (opacity - gt_mask).pow(2)
                v = _add(per_msk, lam_mse)
                if v is not None: f_pix = f_pix + v

        return f_pix

    def training_step(self, batch, batch_idx):
        loss = 0.
        full_batch = batch

        #Volumetric loss
        if self.C(self.config.system.lambda_volumetric) > 0:
            volumetric_loss = 0.
            batch = full_batch['vol_batch']
            out = self.model.render_rays(batch['rays'], batch['background'])

            # update train_num_rays
            if self.config.model.dynamic_ray_sampling:
                # num_samples can be 0 when the geometry is (transiently) empty --
                # no ray hits occupied space. That is a survivable state: the mesh
                # branch still supervises and the surface usually regrows. Dividing
                # by it killed the run outright (ZeroDivisionError), so keep the
                # current ray budget for this step instead.
                _ns = out['num_samples'].sum().detach().item()
                if _ns > 0:
                    train_num_rays = int(self.train_num_rays * (self.train_num_samples / _ns))
                    self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

            _rv = out['rays_valid'][..., 0]
            loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][_rv], batch['rgb'][_rv])
            self.log('train/loss_rgb_mse', loss_rgb_mse)
            volumetric_loss += loss_rgb_mse * self.C(self.config.system.volumetric_loss.lambda_rgb_mse)

            loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][_rv], batch['rgb'][_rv])
            self.log('train/loss_rgb', loss_rgb_l1)
            volumetric_loss += loss_rgb_l1 * self.C(self.config.system.volumetric_loss.lambda_rgb_l1)

            if self.C(self.config.system.volumetric_loss.get("lambda_rgb_l1_adaptive" ,0.0)) > 0:
                pred_rgb = out['comp_rgb_full'][out['rays_valid'][..., 0]]
                gt_rgb = batch['rgb'][out['rays_valid'][..., 0]]
                pixel_errors = F.l1_loss(pred_rgb, gt_rgb, reduction='none')
                gt_intensity = gt_rgb.mean(dim=-1, keepdim=True)
                specular_weights = (1.0 - gt_intensity**5)
                specular_weights = torch.clamp(specular_weights, min=0.01)
                loss_rgb_adaptive = (pixel_errors * specular_weights).mean()
                self.log('train/loss_rg_adaptive', loss_rgb_adaptive)
                volumetric_loss += loss_rgb_adaptive * self.C(self.config.system.volumetric_loss.get("lambda_rgb_l1_adaptive" ,0.0))

            if self.C(self.config.system.volumetric_loss.lambda_eikonal) > 0:
                loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
                self.log('train/loss_eikonal', loss_eikonal)
                volumetric_loss += loss_eikonal * self.C(self.config.system.volumetric_loss.lambda_eikonal)

            opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
            loss_mask_bce = binary_cross_entropy(opacity, batch['fg_mask'].float())
            self.log('train/loss_mask_bce', loss_mask_bce)
            volumetric_loss += loss_mask_bce * (self.C(self.config.system.volumetric_loss.lambda_mask_bce) if self.dataset.has_mask else 0.0)

            loss_mask_mse = F.mse_loss(out['opacity'].squeeze(-1), batch['fg_mask'].float())
            self.log('train/loss_mask_mse', loss_mask_mse)
            volumetric_loss += loss_mask_mse * (self.C(self.config.system.volumetric_loss.lambda_mask_mse) if self.dataset.has_mask else 0.0)

            loss_opaque = binary_cross_entropy(opacity, opacity)
            self.log('train/loss_opaque', loss_opaque)
            volumetric_loss += loss_opaque * self.C(self.config.system.volumetric_loss.lambda_opaque)

            loss_sparsity = torch.exp(-self.config.system.volumetric_loss.sparsity_scale * out['sdf_samples'].abs()).mean()
            self.log('train/loss_sparsity', loss_sparsity)
            volumetric_loss += loss_sparsity * self.C(self.config.system.volumetric_loss.lambda_sparsity)

            if self.C(self.config.system.volumetric_loss.lambda_curvature) > 0:
                assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
                loss_curvature = out['sdf_laplace_samples'].abs().mean()
                self.log('train/loss_curvature', loss_curvature)
                volumetric_loss += loss_curvature * self.C(self.config.system.volumetric_loss.lambda_curvature)

            # distortion loss proposed in MipNeRF360
            # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
            if self.C(self.config.system.volumetric_loss.lambda_distortion) > 0:
                loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
                self.log('train/loss_distortion', loss_distortion)
                volumetric_loss += loss_distortion * self.C(self.config.system.volumetric_loss.lambda_distortion)

            if self.config.model.background.learned and self.C(self.config.system.volumetric_loss.lambda_distortion_bg) > 0:
                loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
                self.log('train/loss_distortion_bg', loss_distortion_bg)
                volumetric_loss += loss_distortion_bg * self.C(self.config.system.volumetric_loss.lambda_distortion_bg)

            # ---- Volumetric monocular normal cosine loss ----
            # Same structure as the mesh-side loss, but on the per-ray
            # composited normal `out['comp_normal']`. Mask = GT foreground AND
            # mono valid AND opacity (detached, so the gradient only flows
            # through the cosine — never tries to reduce loss by lowering
            # opacity). World->camera rotation is invariant to world
            # normalization, so no normalization correction is needed.
            lam_mono_vol = self.C(self.config.system.volumetric_loss.get('lambda_mono_normal', 0.0))
            if (lam_mono_vol > 0
                and getattr(self.dataset, 'has_mono_normal', False)
                and 'mono_normal' in batch
                and 'comp_normal' in out):
                R_w2c = batch['w2c'][:, :3, :3]                       # (R, 3, 3)
                n_world = out['comp_normal']                          # (R, 3)
                n_cam = torch.einsum('rij,rj->ri', R_w2c, n_world)
                n_cam = F.normalize(n_cam, dim=-1)
                mono = batch['mono_normal']                           # (R, 3)
                fg = batch['fg_mask'].float()                         # (R,)
                mono_valid = (mono.norm(dim=-1) > 0.5).float()        # (R,)
                opa = out['opacity'].squeeze(-1).detach()             # (R,)
                pix_w = fg * mono_valid * opa
                cos = (n_cam * mono).sum(-1)
                denom = pix_w.sum().clamp_min(1.0)
                loss_vol_mono_normal = ((1.0 - cos) * pix_w).sum() / denom
                self.log('train/loss_volumetric_mono_normal', loss_vol_mono_normal)
                volumetric_loss += loss_vol_mono_normal * lam_mono_vol

            # if self.C(self.config.system.volumetric_loss.get("lambda_normal_smoothness",0)) > 0:
            #     gt_img = batch['rgb'][..., 0:3]
            #     normal = out["comp_normal"]
            #     x_grad = (gt_img[:, 1:, :, :] - gt_img[:, :-1, :, :]).abs().mean(dim = -1)
            #     y_grad = (gt_img[:, :, 1:, :] - gt_img[:, :, :-1, :]).abs().mean(dim = -1)
            #     x_grad = (x_grad - x_grad.min()) / (x_grad.max() - x_grad.min())
            #     y_grad = (y_grad - y_grad.min()) / (y_grad.max() - y_grad.min())
            #     x_weight = (1.0-x_grad).clamp(0,1.) ** 2
            #     y_weight = (1.0-y_grad).clamp(0,1.) ** 2
            #     normal_x = (normal[:, 1:, :, :] - normal[:, :-1, :, :]).abs().mean(dim = -1)
            #     normal_y = (normal[:, :, 1:, :] - normal[:, :, :-1, :]).abs().mean(dim = -1)
            #     normal_loss = (x_weight*normal_x).mean() + (y_weight*normal_y).mean()
            #     self.log('train/loss_normal_smoothness', normal_loss)
            #     mesh_loss += normal_loss * self.C(self.config.system.mesh_loss.lambda_normal_smoothness)

            loss += volumetric_loss * self.C(self.config.system.lambda_volumetric)

        #Mesh loss
        if self.C(self.config.system.lambda_mesh) > 0:
            mesh_loss = 0.
            batch = full_batch['mesh_batch']

            h = batch['rgb'].shape[1]
            w = batch['rgb'].shape[2]
            n = batch['rgb'].shape[0]

            stochastic = self.C(self.config.system.get('stochastic_mesh_rendering', False))
            num_mesh_samples = int(self.C(self.config.system.get('num_mesh_samples', 1)))

            # REINFORCE topology gradient (Term 1 of E[f] decomposition). Opt-in via
            # mesh_loss.lambda_topology > 0; only meaningful in stochastic mode with
            # >= 2 samples (LOO baseline is undefined for N=1).
            lambda_topology = self.C(self.config.system.mesh_loss.get('lambda_topology', 0.0))
            do_topology = stochastic and lambda_topology > 0 and num_mesh_samples >= 2
            pixel_log_p = None

            if stochastic:
                out, pixel_log_p = self.model.render_mesh(
                    mvp = batch['mvp'],
                    view_pos = batch['campos'],
                    resolution = (h, w),
                    background = batch['background'],
                    rays = batch['rays'],
                    sample_sdf=True,
                    num_samples=num_mesh_samples
                )
                # Plain Monte Carlo averaging over samples: weights = 1/N. Combined
                # with the existing `(loss * weights).sum(0).mean()` pattern this is
                # equivalent to loss.mean(), but keeps the per-loss codepaths unchanged.
                weights = torch.ones((num_mesh_samples, n, h, w, 1), device=out['rgb'].device) / num_mesh_samples
            else:
                out = self.model.render_mesh(
                    mvp = batch['mvp'],
                    view_pos = batch['campos'],
                    resolution = (h, w),
                    background = batch['background'],
                    rays = batch['rays'],
                    sample_sdf = False,
                )
                for k in out.keys():
                    if isinstance(out[k], torch.Tensor):
                        out[k] = out[k].unsqueeze(0)
                weights = torch.ones((1, n, h, w, 1), device=out['rgb'].device)

            gt_rgb = batch['rgb'][..., 0:3].unsqueeze(0).expand_as(out['rgb'][..., 0:3])

            # Per-step mesh coverage. The collapse at ~step 1210 is only visible
            # in validation opacity, which runs every 400 steps and so cannot say
            # whether the surface dies before or after the loss spikes. This is
            # the same quantity, logged every step for ~free.
            if out.get('opacity', None) is not None:
                self.log('train/mesh_opa', out['opacity'].detach().float().mean(),
                         prog_bar=True)

            if self.config.model.get('appearance_embeddings', False):
                img_idx = batch['image_idx'].unsqueeze(0).expand(out['rgb'].shape[0], -1)
                loss_mesh_rgb_mse = MSE_loss_appearance(out['rgb'][...,0:3], gt_rgb, self.model, img_idx, weights=weights)
                self.log('train/loss_mesh_rgb_mse', loss_mesh_rgb_mse)
                mesh_loss += loss_mesh_rgb_mse * self.C(self.config.system.mesh_loss.lambda_rgb_mse)

                loss_mesh_rgb_l1 = L1_loss_appearance(out['rgb'][...,0:3], gt_rgb, self.model, img_idx, weights=weights)
                self.log('train/loss_mesh_rgb_l1', loss_mesh_rgb_l1)
                mesh_loss += loss_mesh_rgb_l1 * self.C(self.config.system.mesh_loss.lambda_rgb_l1)

            else:
                loss_mesh_rgb_mse = (F.mse_loss(out['rgb'][...,0:3], gt_rgb, reduction='none') * weights).sum(0).mean()
                self.log('train/loss_mesh_rgb_mse', loss_mesh_rgb_mse)
                mesh_loss += loss_mesh_rgb_mse * self.C(self.config.system.mesh_loss.lambda_rgb_mse)

                if self.C(self.config.system.mesh_loss.get('lambda_rgb_grad_l1', 0)) > 0:
                    pred = out['rgb'][..., 0:3]
                    grad_x = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
                    grad_y = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]

                    gt_grad_x = gt_rgb[:, :, :, 1:, :] - gt_rgb[:, :, :, :-1, :]
                    gt_grad_y = gt_rgb[:, :, 1:, :, :] - gt_rgb[:, :, :-1, :, :]

                    w_x = weights[:, :, :, 1:, :]
                    w_y = weights[:, :, 1:, :, :]

                    loss_mesh_rgb_grad_l1 = 0.5 * ((F.l1_loss(grad_x, gt_grad_x, reduction='none') * w_x).sum(0).mean() + 
                                                   (F.l1_loss(grad_y, gt_grad_y, reduction='none') * w_y).sum(0).mean())
                    self.log('train/loss_mesh_rgb_grad_l1', loss_mesh_rgb_grad_l1)
                    mesh_loss += loss_mesh_rgb_grad_l1 * self.C(self.config.system.mesh_loss.lambda_rgb_grad_l1)

                loss_mesh_rgb_l1 = (F.l1_loss(out['rgb'][...,0:3], gt_rgb, reduction='none') * weights).sum(0).mean()
                self.log('train/loss_mesh_rgb_l1', loss_mesh_rgb_l1)
                mesh_loss += loss_mesh_rgb_l1 * self.C(self.config.system.mesh_loss.lambda_rgb_l1)

                if self.C(self.config.system.volumetric_loss.get("lambda_rgb_l1_adaptive" ,0.0)) > 0:
                    pred_rgb = out['rgb'][..., 0:3]
                    pixel_errors = F.l1_loss(pred_rgb, gt_rgb, reduction='none')
                    gt_intensity = gt_rgb.mean(dim=-1, keepdim=True)
                    specular_weights = (1.0 - gt_intensity**5)
                    specular_weights = torch.clamp(specular_weights, min=0.01)
                    loss_rgb_adaptive = (pixel_errors * specular_weights * weights).sum(0).mean()
                    self.log('train/loss_mesh_rg_adaptive', loss_rgb_adaptive)
                    mesh_loss += loss_rgb_adaptive * self.C(self.config.system.volumetric_loss.get("lambda_rgb_l1_adaptive" ,0.0))


            # ---- distil the VOLUMETRIC normals into the MESH normals ----------
            # The volumetric branch integrates the SDF gradient over a whole ray,
            # so its normals are typically smoother/better than the rasterised
            # mesh normals early on. Supervise the mesh with them: the volumetric
            # side is rendered under no_grad (pure teacher, no gradient path back
            # into it), and only masked pixels where the volumetric ray actually
            # hit the surface contribute.
            lam_nfv = self.C(self.config.system.mesh_loss.get('lambda_normal_from_volumetric', 0.0))
            if lam_nfv > 0 and 'rays' in batch:
                n_pix = int(self.config.system.mesh_loss.get('normal_from_volumetric_rays', 4096))
                fgm = batch['fg_mask'].reshape(-1)                       # (n*h*w,)
                cand = torch.nonzero(fgm > 0.5, as_tuple=True)[0]
                if cand.numel() > 0:
                    sel = cand[torch.randint(0, cand.numel(), (min(n_pix, cand.numel()),),
                                             device=cand.device)]
                    rays_sel = batch['rays'].reshape(-1, batch['rays'].shape[-1])[sel]
                    bg_sel = batch['background'].reshape(-1, 3)[sel]
                    was_training = self.model.training
                    with torch.no_grad():
                        vout = self.model.render_rays(rays_sel, bg_sel)
                    self.model.train(was_training)
                    if 'comp_normal' in vout:
                        n_vol = F.normalize(vout['comp_normal'].detach().float(), dim=-1)
                        # mesh normals: average over the MC samples, same pixels
                        n_mesh = out['normal'].mean(0).reshape(-1, 3)[sel]
                        n_mesh = F.normalize(n_mesh.float(), dim=-1)
                        hit = vout['rays_valid'][..., 0].float() if 'rays_valid' in vout \
                            else torch.ones_like(n_vol[:, 0])
                        if 'opacity' in vout:
                            hit = hit * (vout['opacity'].reshape(-1).detach() > 0.5).float()
                        denom = hit.sum().clamp_min(1.0)
                        nfv_loss = (((1.0 - (n_mesh * n_vol).sum(-1)) * hit).sum() / denom)
                        self.log('train/loss_normal_from_volumetric', nfv_loss)
                        self.log('train/nfv_rays_hit', hit.sum())
                        mesh_loss += nfv_loss * lam_nfv

            if self.C(self.config.system.mesh_loss.get("lambda_normal_smoothness",0)) > 0:
                normal = out["geometric_normal"]
                x_grad = (gt_rgb[:, :, :, 1:, :] - gt_rgb[:, :, :, :-1, :]).abs().mean(dim=-1, keepdim=True)
                y_grad = (gt_rgb[:, :, 1:, :, :] - gt_rgb[:, :, :-1, :, :]).abs().mean(dim=-1, keepdim=True)
                #Normalize x_grad
                x_grad = (x_grad - x_grad.min()) / (x_grad.max() - x_grad.min() + 1e-7)
                y_grad = (y_grad - y_grad.min()) / (y_grad.max() - y_grad.min() + 1e-7) 
                x_weight = (1.0-x_grad).clamp(0,1.) ** 2
                y_weight = (1.0-y_grad).clamp(0,1.) ** 2
                
                normal_x = (normal[:, :, :, 1:, :] - normal[:, :, :, :-1, :]).abs().mean(dim=-1, keepdim=True)
                normal_y = (normal[:, :, 1:, :, :] - normal[:, :, :-1, :, :]).abs().mean(dim=-1, keepdim=True)   
                
                w_x = weights[:, :, :, 1:, :]
                w_y = weights[:, :, 1:, :, :]
                
                normal_loss = (x_weight * normal_x * w_x).sum(0).mean() + (y_weight * normal_y * w_y).sum(0).mean()
                self.log('train/loss_normal_smoothness', normal_loss)
                mesh_loss += normal_loss * self.C(self.config.system.mesh_loss.get("lambda_normal_smoothness",0))
                # image_weight = (1.0 - get_img_grad_weight(batch['rgb'][..., 0:3]))
                # image_weight = (image_weight).clamp(0,1).detach() ** 2
                # normal_loss = image_weight * (((depth_normal - normal)).abs().sum(0)).mean()
                # mesh_loss += normal_loss * self.C(self.config.system.mesh_loss.get("lambda_single_view_regularization",0))

            # ---- Monocular normal cosine loss ----
            # Compares the rendered geometric normal (world space) to a mono
            # normal predictor's per-image output (camera space). We rotate the
            # rendered normal into camera space via w2c[:3,:3] — this is invariant
            # to world normalization (scale_mat / normalize_poses both rotate the
            # world AND the cameras together), so the dataset-side mono normals
            # do not need any normalization correction.
            lam_mono = self.C(self.config.system.mesh_loss.get('lambda_mono_normal', 0.0))
            if (lam_mono > 0
                and getattr(self.dataset, 'has_mono_normal', False)
                and 'mono_normal' in batch
                and 'geometric_normal' in out):
                R_w2c = batch['w2c'][:, :3, :3]                    # (N, 3, 3)
                n_world = out['geometric_normal']                  # (S, N, H, W, 3)
                if bool(self.config.system.get('mono_normal_orient_fix', True)):
                    # Marching-tets winding is NOT consistently outward (measured
                    # ~54%/46% on trained meshes), so geometric_normal's sign is
                    # arbitrary per face and the plain cosine pushes half the
                    # surface the WRONG way (this is why the loss never helped).
                    # Mono normals are camera-facing by construction: flip the
                    # rendered normal toward the camera (per-pixel view dir,
                    # positions detached) before comparing.
                    cam_o = batch['c2w'][:, :3, 3]                 # (N, 3)
                    vdir = F.normalize(
                        cam_o[None, :, None, None, :] - out['gb_pos'].detach(), dim=-1)
                    sgn = torch.sign((n_world * vdir).sum(-1, keepdim=True))
                    n_world = n_world * torch.where(sgn == 0, torch.ones_like(sgn), sgn)
                n_cam = torch.einsum('nij,snhwj->snhwi', R_w2c, n_world)
                n_cam = F.normalize(n_cam, dim=-1)
                mono = batch['mono_normal'].unsqueeze(0).expand_as(n_cam)
                # Mask: GT foreground AND mono normal is valid AND the mesh
                # actually rasterized geometry at this pixel. The opacity
                # gate is detached so the gradient flows only through the
                # cosine term (refining the orientation of *existing*
                # triangles), not into spawning new geometry where there is
                # none — that path is dominated by photometric/mask losses.
                fg = batch['fg_mask'].float().unsqueeze(0)         # (1, N, H, W)
                mono_valid = (mono.norm(dim=-1) > 0.5).float()     # (S, N, H, W)
                opa = out['opacity'].squeeze(-1).detach()          # (S, N, H, W)
                pix_w = fg * mono_valid * opa * weights.squeeze(-1)
                cos = (n_cam * mono).sum(-1)                       # (S, N, H, W)
                denom = pix_w.sum().clamp_min(1.0)
                loss_mono_normal = ((1.0 - cos) * pix_w).sum() / denom
                self.log('train/loss_mesh_mono_normal', loss_mono_normal)
                mesh_loss += loss_mono_normal * lam_mono

            if self.C(self.config.system.mesh_loss.get("lambda_multiview",0.0)) > 0:
                out_mv = {k: v[0] if isinstance(v, torch.Tensor) and v.ndim >= 4 else v for k, v in out.items()}
                valid_cams = torch.tensor([len(batch['nearest_cam_ids'][i]) != 0 for i in range(len(batch['nearest_cam_ids']))], device=out_mv['rgb'].device)
                nearest_cam_ids = [random.sample(batch['nearest_cam_ids'][i],1)[0] for i in range(len(batch['nearest_cam_ids'])) if valid_cams[i]]
                nearest_cam_batch = self.get_image_batch(torch.tensor(nearest_cam_ids, device=self.dataset.all_images.device), include_rays = True, bg_color = self.config.system.train_background_color)
                #Render depth from closest cam
                nearest_cam_depths = self.model.render_mesh_depth(
                    mvp = nearest_cam_batch['mvp'], 
                    view_pos = nearest_cam_batch['campos'],
                    resolution = (h, w)
                    )

                ix, iy = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy') #shape (h, w)
                pixels = torch.stack([ix, iy], dim=-1).float().to(out_mv['rgb'].device) #shape (h, w, 2)
                pixels = pixels.unsqueeze(0).expand(n, -1, -1, -1)  #shape (n, h, w, 2)

                global_points = out_mv['gb_pos'][valid_cams]                                      # (N,H,W,3)
                ones = torch.ones_like(global_points[..., :1])                      # (N,H,W,1)
                global_points = torch.cat([global_points, ones], dim=-1)            # (N,H,W,4)
                w2c_n = nearest_cam_batch['w2c']                       # (N,4,4)
                c2w_n = nearest_cam_batch['c2w']                       # (N,4,4)
                w2c_v = batch['w2c'][valid_cams]                       # (N,4,4)
                c2w_v = batch['c2w'][valid_cams]                       # (
                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad():
                        points_nearest_cam = torch.einsum(
                            'bij, bhwj -> bhwi',
                            w2c_n,
                            global_points
                        )                
                        points_nearest_cam = points_nearest_cam[..., :3]
                        fx = self.dataset.fx
                        fy = self.dataset.fy
                        cx = self.dataset.cx
                        cy = self.dataset.cy
                        pts_nearest_projections = torch.stack(
                                    [points_nearest_cam[...,0] * fx / -points_nearest_cam[...,2] + cx,
                                    points_nearest_cam[...,1] * fy / points_nearest_cam[...,2] + cy], -1).float()  # (N,H,W,2)
                        d_mask = (pts_nearest_projections[..., 0] > 0) & (pts_nearest_projections[..., 0] < w) &\
                            (pts_nearest_projections[..., 1] > 0) & (pts_nearest_projections[..., 1] < h) & (points_nearest_cam[...,2] < -0.1)
                        pts_nearest_projections[..., 0] /= ((w - 1) / 2)
                        pts_nearest_projections[..., 1] /= ((h - 1) / 2)
                        pts_nearest_projections -= 1
                        map_z = torch.nn.functional.grid_sample(input=nearest_cam_depths.view(n, 1, h, w),
                                                                grid=pts_nearest_projections,
                                                                mode='bilinear',
                                                                padding_mode='border',
                                                                align_corners=True
                                                                )
                        
                        #points_nearest_cam = points_nearest_cam/(points_nearest_cam[...,2:3])
                        points_nearest_cam = points_nearest_cam/torch.norm(points_nearest_cam, dim=-1, keepdim=True)
                        points_nearest_cam = points_nearest_cam * map_z.squeeze(1)[...,None]
                        ones = torch.ones_like(points_nearest_cam[..., :1])                      # (N,H,W,1)
                        points_nearest_cam = torch.cat([points_nearest_cam, ones], dim=-1)  # (N,H,W,4)
                        rel_trans = w2c_v @ c2w_n  # Shape: (B, 4, 4)
                        points_in_view_cam = torch.einsum('bij, bhwj -> bhwi', rel_trans, points_nearest_cam) # (N,H,W,4)
                        pts_view_projections = torch.stack(
                                    [points_in_view_cam[...,0] * fx / -points_in_view_cam[...,2] + cx,
                                    points_in_view_cam[...,1] * fy / points_in_view_cam[...,2] + cy], -1).float()  # (N,H,W,2)
                        pixel_noise = torch.norm(pts_view_projections - pixels.reshape(*pts_view_projections.shape), dim=-1) # (N,H,W)
                        pixel_noise_th = 2.0
                        d_mask = d_mask & (pixel_noise < pixel_noise_th)
                        weights = (1.0 / torch.exp(pixel_noise)).detach()
                        weights[~d_mask] = 0  # Shape : (N,H,W)


                        d_mask_flat = d_mask.reshape(n, -1) # Shape : (N, H*W)
                        weights_flat = weights.reshape(n, -1) # Shape : (N, H*W)

                    if d_mask.sum() > 0:
                        with torch.no_grad():
                            # d_mask = d_mask.reshape(-1)
                            # valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            sample_num = 51200
                            # if d_mask.sum() > sample_num:
                            #     index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                            #     valid_indices = valid_indices[index]
                            # weights = weights.reshape(-1)[valid_indices] 

                            batch_indices_list = []
                            for i in range(n):
                                valid_indices = torch.nonzero(d_mask_flat[i]).squeeze(-1)
                                num_valid = valid_indices.numel()
                                if num_valid >= sample_num:
                                    perm = torch.randperm(num_valid, device=d_mask.device)[:sample_num]
                                    selected_indices = valid_indices[perm]
                                elif num_valid > 0:
                                    idx = torch.randint(0, num_valid, (sample_num,), device=d_mask.device)
                                    selected_indices = valid_indices[idx]
                                else:
                                    selected_indices = torch.zeros(sample_num, dtype=torch.long, device=d_mask.device)
                                batch_indices_list.append(selected_indices)

                            sampled_indices = torch.stack(batch_indices_list)  # (N, sample_num)
                            sampled_weights = torch.gather(weights_flat, 1, sampled_indices) # (N, sample_num)
                            sampled_pixels = torch.gather(pixels.reshape(n, -1, 2), 1, sampled_indices.unsqueeze(-1).expand(-1, -1, 2))  # (N, sample_num, 2)
                            patch_size = 3
                            total_patch_size = (2*patch_size+1)**2
                            offsets = torch.arange(-patch_size, patch_size + 1, device=pixels.device) #shape (2*patch_size+1,)
                            offsets = torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2) #shape (1, (2*patch_size+1)^2, 2)
                            ori_pixels_patch = sampled_pixels.reshape(n, -1, 1, 2) + offsets.float() #shape (N, sample_num, (2*patch_size+1)^2, 2)
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, :, 0] = 2 * pixels_patch[:, :, :, 0] / (w - 1) - 1.0
                            pixels_patch[:, :, :, 1] = 2 * pixels_patch[:, :, :, 1] / (h - 1) - 1.0
                            gt_gray_image = batch['grayscale'][valid_cams]  # (N,H,W)
                            ref_gray_val = F.grid_sample(gt_gray_image.unsqueeze(1), pixels_patch, align_corners=True) # (N,1,sample_num,(2*patch_size+1)^2)

                            # 1. Compute Relative Pose (Ref -> Nearest)
                            # rel_trans is (Nearest -> Ref). We need the inverse (Ref -> Nearest) for H computation.
                            T_ref2nearest = torch.inverse(rel_trans.float()) # Shape: (N, 4, 4)
                            R = T_ref2nearest[:, :3, :3] # (N, 3, 3)
                            t = T_ref2nearest[:, :3, 3]  # (N, 3)

                        ref_local_n = out_mv['geometric_normal'][valid_cams]  # (N,H,W,3)
                        ref_local_d = out_mv['depth'][valid_cams]  # (N,H,W,1)


                        # 2. Construct Intrinsic Matrices (K)
                        # Construct K for the Reference View (current view)
                        # Assuming fx, fy, cx, cy are available from context (as used in projection logic above)
                        K_ref = torch.zeros((n, 3, 3), device=pixels.device)
                        K_ref[:, 0, 0] = fx
                        K_ref[:, 1, 1] = fy
                        K_ref[:, 0, 2] = cx
                        K_ref[:, 1, 2] = cy
                        K_ref[:, 2, 2] = 1.0

                        # Construct K for the Nearest View (source view)
                        # extracting from nearest_cam_batch, assuming similar key structure
                        K_near = K_ref

                        # 3. Flatten Geometry for Broadcasting
                        # Flatten Normals: (N, H, W, 3) -> (N, H*W, 3, 1)
                        rot_w2c = w2c_v[:, :3, :3]
                        n_world_flat = ref_local_n.reshape(n, -1, 3) 
                        #n_view_flat = torch.matmul(rot_w2c.unsqueeze(1), n_world_flat.unsqueeze(-1))
                        n_view_flat = n_world_flat.unsqueeze(-1)
                        # Flatten Depth: (N, H, W, 1) -> (N, H*W, 1, 1)
                        d_flat = ref_local_d.reshape(n, -1, 1).unsqueeze(-1)


                        indices_n = sampled_indices.view(n, sample_num, 1, 1).expand(-1, -1, 3, 1)
                        n_view_flat = torch.gather(n_view_flat, 1, indices_n) # (N, sample_num, 3, 1)
                        indices_d = sampled_indices.view(n, sample_num, 1, 1).expand(-1, -1, 1, 1)
                        d_flat = torch.gather(d_flat, 1, indices_d) # (N, sample_num, 1, 1)


                        # 4. Compute Geometric Homography (Plane Induced)
                        # Formula: H_geom = R - (t @ n.T) / d
                        # Dimensions: 
                        # R: (N, 1, 3, 3)
                        # t: (N, 1, 3, 1)
                        # n.transpose: (N, HW, 1, 3)
                        # term: (t @ n.T) is (N, HW, 3, 3)
                        
                        R_exp = R.unsqueeze(1) # Broadcast R to all pixels
                        t_exp = t.unsqueeze(1).unsqueeze(-1) # (N, 1, 3, 1)
                        tn_T = torch.matmul(t_exp, n_view_flat.transpose(-2, -1)) # Outer product t * n^T
                        
                        # Add epsilon to depth to avoid division by zero
                        H_geom = R_exp - tn_T / (d_flat + 1e-7)

                        # 5. Composite Final Homography with Intrinsics
                        # H = K_near @ H_geom @ K_ref_inv
                        K_ref_inv = torch.inverse(K_ref)
                        
                        # Broadcast Ks to (N, 1, 3, 3)
                        H_ref_to_neareast = torch.matmul(
                            K_near.unsqueeze(1), 
                            torch.matmul(H_geom, K_ref_inv.unsqueeze(1))
                        )
                        patches_flat = ori_pixels_patch.reshape(-1, ori_pixels_patch.shape[-2], 2)  # (N*sample_num, (2*patch_size+1)^2, 2)
                        H_flat = H_ref_to_neareast.reshape(-1, 3, 3) # (N*sample_num, 3, 3)
                        transformed_patches_flat = patch_warp(H_flat, patches_flat)  # (N*sample_num, (2*patch_size+1)^2, 2)
                        transformed_patches_flat[..., 0] = 2 * transformed_patches_flat[..., 0] / (w - 1) - 1.0
                        transformed_patches_flat[..., 1] = 2 * transformed_patches_flat[..., 1] / (h - 1) - 1.0
                        sampling_grid = transformed_patches_flat.reshape(n, -1, transformed_patches_flat.shape[1], 2) # (N, sample_num, (2*patch_size+1)^2, 2)
                        nearest_image_gray = nearest_cam_batch['grayscale']  # (N,H,W)
                        sampled_gray_val = F.grid_sample(nearest_image_gray.unsqueeze(1), sampling_grid, align_corners=True, padding_mode='border') # (N,1,sample_num,(2*patch_size+1)^2)

                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val.view(-1,total_patch_size), sampled_gray_val.view(-1,total_patch_size))
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * sampled_weights.reshape(-1)
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss =  ncc.mean()
                            self.log('train/loss_mesh_ncc', ncc_loss)
                            mesh_loss += (ncc_loss * self.C(self.config.system.mesh_loss.get("lambda_multiview",0.0)))
                    
                # map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                # pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                # pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                # R = torch.tensor(nearest_cam.R).float().cuda()
                # T = torch.tensor(nearest_cam.T).float().cuda()
                # pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                # pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                # pts_projections = torch.stack(
                #             [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                #             pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                # pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                
                #Compute multiview consistency factor/mask

                #Compute local ghomographyts
                #Compute patch loss

            if self.C(self.config.system.mesh_loss.get("lambda_ssim",0)) > 0:
                S_, n_, h_, w_, _ = out['rgb'][..., 0:3].shape
                pred_flat = out['rgb'][..., 0:3].reshape(S_*n_, h_, w_, 3).permute(0, 3, 1, 2).contiguous()
                gt_flat = gt_rgb.reshape(S_*n_, h_, w_, 3).permute(0, 3, 1, 2).contiguous()
                loss_mesh_ssim = 1.0 - ssim(pred_flat, gt_flat)
                self.log('train/loss_mesh_ssim', loss_mesh_ssim)
                mesh_loss += loss_mesh_ssim * self.C(self.config.system.mesh_loss.lambda_ssim)

            if self.C(self.config.system.mesh_loss.lambda_eikonal) > 0:
                if out.get('sdf_grad_samples', None) is not None:
                    loss_mesh_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
                    self.log('train/loss_mesh_eikonal', loss_mesh_eikonal)
                    mesh_loss += loss_mesh_eikonal * self.C(self.config.system.mesh_loss.lambda_eikonal)

            if self.C(self.config.system.mesh_loss.lambda_angles) > 0:
                loss_mesh_angles = self.model.mesh_angle_loss()
                self.log('train/loss_mesh_angles', loss_mesh_angles)
                mesh_loss += loss_mesh_angles * self.C(self.config.system.mesh_loss.lambda_angles)

            # Mesh-quality regularizers (anti-sliver). Extract the mesh once and share
            # it across the triangle-based terms; the tet term uses the grid directly.
            _ml = self.config.system.mesh_loss
            lam_tq = self.C(_ml.get('lambda_triangle_quality', 0.0))
            lam_es = self.C(_ml.get('lambda_edge_size', 0.0))
            lam_lap = self.C(_ml.get('lambda_laplacian', 0.0))
            lam_sa = self.C(_ml.get('lambda_small_area', 0.0))
            if lam_tq > 0 or lam_es > 0 or lam_lap > 0 or lam_sa > 0:
                _reg_mesh = self.model.getMesh()
                if lam_sa > 0:
                    l_sa = self.model.mesh_small_area_loss(_reg_mesh)
                    self.log('train/loss_small_area', l_sa); mesh_loss += l_sa * lam_sa
                if lam_tq > 0:
                    l_tq = self.model.mesh_triangle_quality_loss(_reg_mesh)
                    self.log('train/loss_tri_quality', l_tq); mesh_loss += l_tq * lam_tq
                if lam_es > 0:
                    l_es = self.model.mesh_edge_size_loss(_reg_mesh)
                    self.log('train/loss_edge_size', l_es); mesh_loss += l_es * lam_es
                if lam_lap > 0:
                    l_lap = self.model.mesh_laplacian_loss(_reg_mesh)
                    self.log('train/loss_laplacian', l_lap); mesh_loss += l_lap * lam_lap
            lam_tet = self.C(_ml.get('lambda_tet_quality', 0.0))
            if lam_tet > 0:
                l_tet = self.model.tet_quality_loss()
                self.log('train/loss_tet_quality', l_tet); mesh_loss += l_tet * lam_tet

            if 'opacity' in out.keys():
                opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3) # (S, N, H, W)
                gt_mask = batch['fg_mask'].float().unsqueeze(0).expand_as(opacity) # (S, N, H, W)
                w_mask = weights.squeeze(-1) # (S, N, H, W)

                if self.C(self.config.system.mesh_loss.lambda_mask_bce) > 0:
                    loss_mesh_mask_bce = (F.binary_cross_entropy(opacity, gt_mask, reduction='none') * w_mask).sum(0).mean()
                    self.log('train/loss_mesh_mask', loss_mesh_mask_bce)
                    mesh_loss += loss_mesh_mask_bce * (self.C(self.config.system.mesh_loss.lambda_mask_bce) if self.dataset.has_mask else 0.0)

                if self.C(self.config.system.mesh_loss.lambda_mask_mse) > 0:
                    loss_mesh_mask_mse = (F.mse_loss(opacity, gt_mask, reduction='none') * w_mask).sum(0).mean()
                    self.log('train/loss_mesh_mask_mse', loss_mesh_mask_mse)
                    mesh_loss += loss_mesh_mask_mse * (self.C(self.config.system.mesh_loss.lambda_mask_mse) if self.dataset.has_mask else 0.0)

            if 'kd_grad' in out.keys():
                loss_mesh_albedo  =  (out["kd_grad"][..., :-1] * out["kd_grad"][..., -1:]).sum(0).mean()
                self.log('train/loss_mesh_albedo', loss_mesh_albedo)
                mesh_loss += loss_mesh_albedo * self.C(self.config.system.mesh_loss.lambda_albedo)

            if 'occlusion' in out.keys():
                loss_mesh_visibility =  (out["occlusion"][..., :-1] * out["occlusion"][..., -1:]).sum(0).mean()
                self.log('train/loss_mesh_visibility', loss_mesh_visibility)
                mesh_loss += loss_mesh_visibility * self.C(self.config.system.mesh_loss.lambda_visibility)

            # ---- REINFORCE topology gradient (Term 1) ----
            # Decomposing E_s[f(s)] = sum_tau P(tau|mu,sigma) * E[f|tau], the
            # pathwise term (Term 2) is what autograd through DMTet+rasterize
            # already gives via the existing per-loss accumulations above. The
            # missing piece is the topology gradient, estimated by a score-
            # function trick with a leave-one-out baseline (variance reduction)
            # and visible-tet locality (Rao-Blackwellization).
            if do_topology and pixel_log_p is not None:
                if self.config.model.get('appearance_embeddings', False):
                    # The appearance-embedding losses don't decompose into a
                    # clean per-pixel form, so the score-function advantage
                    # would be biased. Disable here and warn.
                    print("[radtets-system] WARNING: lambda_topology > 0 with "
                          "appearance_embeddings is unsupported; skipping the "
                          "topology surrogate this step.")
                else:
                    with torch.no_grad():
                        f_pix = self._build_mesh_f_pix(out, gt_rgb, batch)  # (N, B, H, W)
                        N = f_pix.shape[0]
                        total = f_pix.sum(dim=0, keepdim=True)
                        loo_mean = (total - f_pix) / (N - 1)
                        advantage = (f_pix - loo_mean)
                    # log_p shape (N, B, H, W, 1); align dims and aggregate.
                    score_loss = (advantage * pixel_log_p.squeeze(-1)).mean()
                    self.log('train/loss_mesh_topology_score', score_loss)
                    self.log('train/topology_advantage_abs',
                             advantage.abs().mean().detach())
                    self.log('train/topology_log_p_mean',
                             pixel_log_p.detach().mean())
                    mesh_loss += score_loss * lambda_topology

            loss += mesh_loss * self.C(self.config.system.lambda_mesh)
                                                    

        # ---- Normal-texture deviation penalty ----
        # anchors the shading normal to the geometric one so the texture stays
        # a local bump correction instead of drifting into a free latent field.
        # Evaluated on DETACHED mesh surface positions -> its own fresh graph,
        # independent of the render/create_graph graph (avoids double-backward).
        nt = getattr(self.model, 'normal_texture', None)
        lam_ntdev = self.C(self.config.system.get('lambda_normaltex_deviation', 0.0)) if nt is not None else 0.0
        if lam_ntdev > 0 and self.C(self.config.system.lambda_mesh) > 0 and 'gb_pos' in out:
            mask_key = 'hit' if 'hit' in out else ('opacity' if 'opacity' in out else None)
            hit = out[mask_key][..., 0] > 0.5 if mask_key is not None else None
            pos = out['gb_pos']
            pos = pos[hit] if (hit is not None and hit.any()) else pos.reshape(-1, 3)
            loss_ntdev = self.model.normal_texture.deviation_penalty(pos.detach())
            self.log('train/normaltex_deviation', loss_ntdev, prog_bar=True)
            loss += loss_ntdev * lam_ntdev

        if self.C(self.config.system.reg_loss.lambda_sdf) > 0:
            loss_sdf = self.model.improved_sdf_loss()
            self.log('train/loss_sdf', loss_sdf)
            loss += loss_sdf * self.C(self.config.system.reg_loss.lambda_sdf)

        if self.C(self.config.system.reg_loss.get('lambda_sdf_range', 0)) > 0:
            loss_sdf_range = self.model.sdf_range_regularization()
            self.log('train/loss_sdf_range', loss_sdf_range)
            loss += loss_sdf_range * self.C(self.config.system.reg_loss.lambda_sdf_range)

        if self.C(self.config.system.reg_loss.lambda_light) > 0 and self.model.learn_light and self.model.appearance_model == 'pbr':
            loss_light = self.model.lgt.regularizer()
            self.log('train/loss_light', loss_light)
            loss += loss_light * self.C(self.config.system.reg_loss.lambda_light)

        #Check if model has variance attribute
        if hasattr(self.model, 'variance'):
            self.log('train/inv_s', torch.exp(self.model.variance), prog_bar=True)
        #else:
            #self.log('train/mean_inv_s', torch.exp(self.model.initial_variance), prog_bar=True)

        # Log global features from the model
        if hasattr(self.model, 'feature_registry'):
            for name, feat in self.model.feature_registry.items():
                if feat.get('storage') == 'global' and hasattr(self.model, name):
                    val = getattr(self.model, name)
                    if val.numel() == 1:
                        self.log(f'train/{name}', val.item())
                    elif val.numel() <= 16: # Avoid flooding for large vectors
                        for i in range(val.numel()):
                            self.log(f'train/{name}_{i}', val[i].item())

        #self.log('train/num_rays', self.train_num_rays, prog_bar=True)
        self.log('train_params/lambda_volumetric', self.C(self.config.system.lambda_volumetric))
        self.log('train_params/lambda_mesh', self.C(self.config.system.lambda_mesh))

        for name, value in self.config.system.volumetric_loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/volumetric_{name}', self.C(value))

        for name, value in self.config.system.mesh_loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/mesh_{name}', self.C(value))

        return {
            'loss': loss
        }
    
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    
    def validation_step(self, batch, batch_idx):
        #self.model.save_grid(self.get_save_path(f"it{self.global_step}-grid.ply"))
        W, H = self.dataset.img_wh
        image_grid = []
        volumetric_psnr = torch.tensor(0.)
        mesh_psnr = torch.tensor(0.)
        mesh_psnr_fg = torch.tensor(0.)          # foreground-only PSNR (excludes white bg)
        volumetric_psnr_fg = torch.tensor(0.)

        if self.config.system.validate_mesh:
            stochastic = self.C(self.config.system.get('stochastic_mesh_rendering', False))
            num_mesh_samples = min(int(self.C(self.config.system.get('num_mesh_samples', 0))), 3)

            # 1. Standard "Mean" render (expected value)
            buffers = self.model.render_mesh(
                mvp = batch['mvp'], 
                view_pos = batch['campos'],
                resolution = (H, W),
                background = batch['background'],
                rays = batch['rays'],
                sample_sdf = False
            )

            # Save features to a square grid with white padding
            if (self.global_step % self.config.system.get('save_features_interval', 1)) == 0:
                feature_imgs = []
                for key, value in buffers.items():
                    if key.startswith('feature_'):
                        feature_imgs.append(value.view(H, W))  # grayscale single-channel
                num_feats = len(feature_imgs)
                if num_feats > 0:
                    # Determine grid size (smallest square)
                    grid_size = int(math.ceil(num_feats ** 0.5))
                    total_slots = grid_size * grid_size
                    white_img = torch.ones_like(feature_imgs[0])
                    while len(feature_imgs) < total_slots:
                        feature_imgs.append(white_img)

                    # Build grid for save_image_grid
                    feature_img_grid = []
                    for i in range(grid_size):
                        row = []
                        for j in range(grid_size):
                            idx = i * grid_size + j
                            row.append({
                                'type': 'grayscale',
                                'img': torch.sigmoid(feature_imgs[idx]),
                                'kwargs': {'data_range': (0, 1)}
                            })
                        feature_img_grid.append(row)

                    # Save square layout
                    self.save_image_grid(
                        f"it{self.global_step}-{batch['index'][0].item()}-features.png",
                        feature_img_grid
                    )

            if self.config.model.get('appearance_embeddings', False):
                buffers['rgb'] = L1_loss_appearance(buffers['rgb'], batch['rgb'], model = self.model, view_idx = batch['image_idx'], return_transformed_image= True)
            mesh_psnr = self.criterions['psnr'](buffers['rgb'].to(batch['rgb']), batch['rgb'])
            _fg = batch['fg_mask'].reshape(-1) > 0.5
            if _fg.any():
                mesh_psnr_fg = self.criterions['psnr'](buffers['rgb'].reshape(-1, 3).to(batch['rgb']),
                                                       batch['rgb'].reshape(-1, 3), valid_mask=_fg)
            # Diagnostic grid. opacity + depth are added so a collapsing/imploding
            # mesh is unmistakable: opacity goes blank (no geometry rasterized) and
            # depth degenerates. inv_s (the variance that blows up on collapse) is
            # logged so we can correlate the step where it diverges.
            inv_s_val = float(torch.exp(self.model.variance)) if hasattr(self.model, 'variance') else float('nan')
            # 4 panels (must match the volumetric row's panel count or save_image_grid
            # can't stack the rows). gt | rendered rgb | geometric normal | opacity.
            # opacity is the clear collapse indicator: it goes blank when the mesh
            # rasterizes nothing.
            mesh_img_grid = [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': buffers['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': buffers['geometric_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                {'type': 'grayscale', 'img': buffers['opacity'].view(H, W), 'kwargs': {'data_range': (0, 1)}},
            ]
            self.print(f"[val it{self.global_step}] inv_s={inv_s_val:.1f} "
                       f"mesh_opacity_mean={float(buffers['opacity'].mean()):.4f} "
                       f"mesh_psnr={float(mesh_psnr):.2f} mesh_psnr_fg={float(mesh_psnr_fg):.2f}")
            image_grid.append(mesh_img_grid)

            # ---- Normal-texture visualization (only when enabled) ----
            # geometric normal | shading normal | tilt heatmap (white = 18°+).
            if getattr(self.model, 'normal_texture', None) is not None:
                with torch.no_grad():
                    gpos = buffers['gb_pos'].view(H, W, 3)
                    ngeom = F.normalize(buffers['geometric_normal'].view(H, W, 3), dim=-1)
                    nshade = self.model._shading_normals(gpos, ngeom)
                    cos = (nshade * ngeom).sum(-1).clamp(-1.0, 1.0)
                    fg = buffers['opacity'].view(H, W) > 0.5
                    if fg.any():
                        mean_deg = float(torch.rad2deg(torch.acos(cos[fg])).mean())
                        self.log('val/normaltex_mean_deg', mean_deg, rank_zero_only=True)
                        self.print(f"[normaltex it{self.global_step}] mean tilt {mean_deg:.2f} deg")
                self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}-normaltex.png", [[
                    {'type': 'rgb', 'img': ngeom, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                    {'type': 'rgb', 'img': nshade, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                    {'type': 'grayscale', 'img': ((1.0 - cos) * 20.0).clamp(0.0, 1.0), 'kwargs': {'data_range': (0, 1)}},
                ]])

            # ---- Mono-normal debug visualization ----
            # Side-by-side check that what we OPTIMIZE AGAINST (mono target in
            # camera frame) is in the same coordinate convention as what we
            # COMPARE IT TO in the loss (rendered geometric_normal rotated by
            # w2c[:3,:3]). If the two left panels don't roughly match (similar
            # color tinting on visible surfaces), the convention is wrong.
            if getattr(self.dataset, 'has_mono_normal', False) and 'mono_normal' in batch:
                with torch.no_grad():
                    mono_cam = batch['mono_normal'].view(H, W, 3)
                    # Rendered geometric normal: world -> camera (OpenGL) via w2c.
                    R_w2c = batch['w2c'][0, :3, :3]
                    n_world = buffers['geometric_normal'].view(H, W, 3)
                    if bool(self.config.system.get('mono_normal_orient_fix', True)) \
                            and 'gb_pos' in buffers:
                        # same winding-agnostic flip as the training loss
                        cam_o = batch['c2w'][0, :3, 3]
                        vdir = F.normalize(cam_o[None, None, :]
                                           - buffers['gb_pos'].view(H, W, 3), dim=-1)
                        sgn = torch.sign((n_world * vdir).sum(-1, keepdim=True))
                        n_world = n_world * torch.where(sgn == 0, torch.ones_like(sgn), sgn)
                    n_cam = torch.einsum('ij,hwj->hwi', R_w2c, n_world)
                    n_cam = F.normalize(n_cam, dim=-1)
                    # Per-pixel cosine similarity (dot product) on foreground +
                    # valid mono pixels — written out as a grayscale map.
                    fg = batch['fg_mask'].float().view(H, W)
                    valid = (mono_cam.norm(dim=-1) > 0.5) & (fg > 0.5)
                    prod = n_cam * mono_cam                                              # (H, W, 3)
                    cos_map = prod.sum(-1).clamp(-1.0, 1.0)
                    if valid.any():
                        v = valid
                        # Per-axis breakdown of the dot product. If e.g. axis_z is
                        # strongly NEGATIVE while axis_x/y are near zero or positive,
                        # only Z is sign-flipped -> set mono_normal_sign: [1,1,-1].
                        # If all three are negative, the mesh normal is fully
                        # inverted -> mono_normal_sign: [-1,-1,-1].
                        self.log('val/mono_normal_mean_cos',
                                 cos_map[v].mean().item(), rank_zero_only=True)
                        self.log('val/mono_normal_axis_x',
                                 prod[..., 0][v].mean().item(), rank_zero_only=True)
                        self.log('val/mono_normal_axis_y',
                                 prod[..., 1][v].mean().item(), rank_zero_only=True)
                        self.log('val/mono_normal_axis_z',
                                 prod[..., 2][v].mean().item(), rank_zero_only=True)
                        # Also dump the same numbers to stdout for quick eyeballing —
                        # the TB log line is easy to miss in the middle of training.
                        print(f"[mono-debug] step={self.global_step} idx={batch['index'][0].item()} "
                              f"mean_cos={cos_map[v].mean().item():+.3f} "
                              f"axis=[{prod[...,0][v].mean().item():+.3f}, "
                              f"{prod[...,1][v].mean().item():+.3f}, "
                              f"{prod[...,2][v].mean().item():+.3f}]")

                debug_grid = [
                    {'type': 'rgb', 'img': mono_cam,
                     'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},   # mono target (cam frame)
                    {'type': 'rgb', 'img': n_cam,
                     'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},   # rendered normal (cam frame)
                    {'type': 'rgb', 'img': n_world,
                     'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},   # rendered normal (world frame)
                    {'type': 'grayscale', 'img': cos_map,
                     'kwargs': {'data_range': (-1, 1)}},                          # per-pixel agreement
                ]
                image_grid.append(debug_grid)

            # 2. Stochastic renders (if enabled)
            if stochastic and num_mesh_samples > 0:
                stacked_buffers, pixel_probs = self.model.render_mesh(
                    mvp = batch['mvp'], 
                    view_pos = batch['campos'],
                    resolution = (H, W),
                    background = batch['background'],
                    rays = batch['rays'],
                    sample_sdf = True,
                    num_samples = num_mesh_samples
                )
                if self.config.model.get('appearance_embeddings', False):
                    stacked_buffers['rgb'] = L1_loss_appearance(stacked_buffers['rgb'], batch['rgb'], model = self.model, view_idx = batch['image_idx'], return_transformed_image= True)

                for i in range(num_mesh_samples):
                    sample_img_grid = [
                        {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                        {'type': 'rgb', 'img': stacked_buffers['rgb'][i].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                        {'type': 'rgb', 'img': stacked_buffers['geometric_normal'][i].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                        {'type': 'rgb', 'img': stacked_buffers['normal'][i].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
                    ]
                    image_grid.append(sample_img_grid)
        
        if self.config.system.validate_volumetric and self.C(self.config.system.lambda_volumetric) > 0:
            # render_rays needs autograd enabled for the analytic SDF normals even
            # at validation time (requires trainer.inference_mode=false); detach
            # PER CHUNK so at most one chunk's graph is alive at a time.
            def _rr_detached(rays_c, bg_c):
                with torch.enable_grad():
                    o = self.model.render_rays(rays_c, bg_c)
                return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in o.items()}
            out = chunk_batch(_rr_detached, self.validation_ray_chunk, False, batch['rays'], batch['background'].view(-1, 3))
            if self.config.model.get('appearance_embeddings', False):
                out['comp_rgb_full'] = L1_loss_appearance(out['comp_rgb_full'].view(batch['rgb'].shape), batch['rgb'], model = self.model, view_idx = batch['image_idx'], return_transformed_image= True)
            volumetric_psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']).view(-1,3), batch['rgb'].view(-1, 3))
            _fgv = batch['fg_mask'].reshape(-1) > 0.5
            if _fgv.any():
                volumetric_psnr_fg = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']).view(-1,3),
                                                             batch['rgb'].view(-1,3), valid_mask=_fgv)
            volumetric_img_grid = [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}}
            ]
            image_grid.append(volumetric_img_grid)
        


        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", image_grid)
        self.print(f"[val it{self.global_step}] psnr_fg mesh={float(mesh_psnr_fg):.2f} "
                   f"vol={float(volumetric_psnr_fg):.2f} (foreground-only, excludes bg)")
        return {
            'volumetric_psnr': volumetric_psnr,
            'mesh_psnr': mesh_psnr,
            'mesh_psnr_fg': mesh_psnr_fg,
            'volumetric_psnr_fg': volumetric_psnr_fg,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        if (self.global_step % self.config.system.get('save_mesh_interval', 1)) == 0:
            self.model.save_grid(self.get_save_path(f"it{self.global_step}-grid.ply"), scale = self.dataset.scale, transform = torch.linalg.inv(self.dataset.transform))
            self.model.save_mesh(self.get_save_path(f"it{self.global_step}-mesh.ply"), scale = self.dataset.scale, transform = torch.linalg.inv(self.dataset.transform))
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'volumetric_psnr': step_out['volumetric_psnr'], 'mesh_psnr': step_out['mesh_psnr'], 'mesh_psnr_fg': step_out.get('mesh_psnr_fg', torch.tensor(0.))}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'volumetric_psnr': step_out['volumetric_psnr'][oi], 'mesh_psnr': step_out['mesh_psnr'][oi], 'mesh_psnr_fg': step_out['mesh_psnr_fg'][oi]}
            volumetric_psnr = torch.mean(torch.stack([o['volumetric_psnr'] for o in out_set.values()]))
            mesh_psnr = torch.mean(torch.stack([o['mesh_psnr'] for o in out_set.values()]))
            self.log('val/volumetric_psnr', volumetric_psnr, prog_bar=True, rank_zero_only=True)
            self.log('val/mesh_psnr', mesh_psnr, prog_bar=True, rank_zero_only=True)
            # HELD-OUT (novel-view) foreground-masked PSNR: mean over ALL views NOT
            # used for training -- the proper sparse-view eval (not just 2 val batches).
            _tv = set(self.config.model.get('train_view_ids', []) or [])
            _ho = [v['mesh_psnr_fg'] for k, v in out_set.items() if k not in _tv]
            if _ho:
                _hom = torch.mean(torch.stack([torch.as_tensor(x).float() for x in _ho]))
                self.print(f"HELDOUT_PSNR_FG mesh={float(_hom):.3f} n_heldout={len(_ho)}/{len(out_set)} (novel views, fg-only)")

    def on_load_checkpoint(self, checkpoint):
        # The tet grid grows via densification during training, so a freshly-built
        # model has fewer points than a trained checkpoint. Resize every per-point
        # tensor to the checkpoint's count so load_state_dict fits; the triangulation
        # is rebuilt in on_validation_start / on_test_start.
        import torch.nn as nn
        sd = checkpoint.get('state_dict', {})
        for gname in ['tetrahedral_grid', 'grid_manager']:
            grid = getattr(self.model, gname, None)
            if gname == 'grid_manager': grid = getattr(grid, 'grid', None)
            if grid is None or not hasattr(grid, 'primal_points'): continue
            key = f'model.{gname}.grid.primal_points' if gname == 'grid_manager' else f'model.{gname}.primal_points'
            if key not in sd: continue
            N = sd[key].shape[0]
            if grid.primal_points.shape[0] == N: continue
            dev = grid.primal_points.device
            grid.primal_points = nn.Parameter(torch.zeros(N, 3, device=dev))
            for fname in getattr(grid, 'feature_names', []):
                old = getattr(grid, fname)
                setattr(grid, fname, nn.Parameter(torch.zeros(N, old.shape[1], device=dev)))
            if hasattr(grid, 'point_grad_accum'):
                grid.point_grad_accum = torch.zeros(N, device=dev)
            grid.recompute_primal_points_uncontracted = True
            grid._tetrahedra_tracer = None
        self._grid_needs_retriangulate = True

    def _retriangulate_loaded_grid(self):
        if not getattr(self, '_grid_needs_retriangulate', False):
            return
        import radfoam
        import torch.nn as nn
        grid = self.model.tetrahedral_grid
        grid.recompute_primal_points_uncontracted = True
        pu = grid.primal_points_uncontracted
        grid.triangulation = radfoam.Triangulation(pu.contiguous())   # rebuild Delaunay from loaded pts
        # radfoam reorders points to its canonical layout; apply that permutation to
        # BOTH points and per-point features (mirrors initialize_points) so the tet
        # indices line up with the loaded features -- else the SDF is scrambled.
        perm = grid.triangulation.permutation().to(torch.long)
        grid.primal_points = nn.Parameter(grid.primal_points[perm])
        for fname in grid.feature_names:
            setattr(grid, fname, nn.Parameter(getattr(grid, fname)[perm]))
        grid.recompute_primal_points_uncontracted = True
        grid._tetrahedra_tracer = None
        grid.update_triangulation(rebuild=False)                      # recompute indices/edges/tracer
        self._grid_needs_retriangulate = False
        self.print(f"[reload] rebuilt tet grid: {grid.primal_points.shape[0]} points")

    def on_validation_start(self):
        self._retriangulate_loaded_grid()

    def on_test_start(self):
        self._retriangulate_loaded_grid()

    def test_step(self, batch, batch_idx):
        W, H = self.dataset.img_wh
        image_grid = []
        volumetric_psnr = torch.tensor(0.)
        mesh_psnr = torch.tensor(0.)
        # if self.config.system.validate_volumetric:
        #     out = chunk_batch(self.model.render_rays, self.validation_ray_chunk, True, batch['rays'], batch['background'])
        #     volumetric_psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        #     volumetric_img_grid = [
        #         {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #         {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #         {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
        #         {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        #     ]
        #     image_grid.append(volumetric_img_grid)
        if self.config.system.validate_mesh:
            buffers = self.model.render_mesh(
                mvp=batch['mvp'],
                view_pos=batch['campos'],     # match validation: view_pos not campos
                resolution=(H, W),
                background=batch['background'],
                rays=batch['rays'],
                sample_sdf=False,
                num_offset_samples = 1
            )

            if self.config.model.get('appearance_embeddings', False):
                buffers['rgb'] = L1_loss_appearance(
                    buffers['rgb'], batch['rgb'],
                    model=self.model,
                    view_idx=batch['image_idx'],
                    return_transformed_image=True
                )

            mesh_psnr = self.criterions['psnr'](
                buffers['rgb'].to(batch['rgb']),
                batch['rgb']
            )

            mesh_ssim = self.criterions['ssim'](
                buffers['rgb'].cpu().permute(0, 3, 1, 2),
                batch['rgb'].cpu().permute(0, 3, 1, 2)
            )

            mesh_img_grid = [
                {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': buffers['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': buffers['geometric_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
                {'type': 'rgb', 'img': buffers['normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
            ]
            image_grid.append(mesh_img_grid)

        # --- Match validation step file naming ---
        self.save_image_grid(
            f"test/it{self.global_step}-test{batch['index'][0].item()}.png",
            image_grid
        )

        return {
            'volumetric_psnr': volumetric_psnr,
            'mesh_psnr': mesh_psnr,
            'mesh_ssim': mesh_ssim,
            'index': batch['index']
        }
        
        # if self.config.system.validate_mesh:
        #     buffers = self.model.render_mesh(
        #         mvp = batch['mvp'], 
        #         campos = batch['campos'],
        #         resolution = (H, W),
        #         background = batch['background'],
        #         rays = batch['rays'],
        #         sample_sdf = False
        #     )
        #     mesh_psnr = self.criterions['psnr'](buffers['rgb'].to(batch['rgb']), batch['rgb'])
        #     mesh_img_grid = [
        #         {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #         {'type': 'rgb', 'img': buffers['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #         {'type': 'grayscale', 'img': buffers['depth'].view(H, W), 'kwargs': {}},
        #         {'type': 'rgb', 'img': buffers['normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        #     ]
        #     image_grid.append(mesh_img_grid)

        # self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", image_grid)
        # return {
        #     'volumetric_psnr': volumetric_psnr,
        #     'mesh_psnr': mesh_psnr,
        #     'index': batch['index']
        # }
          
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'volumetric_psnr': step_out['volumetric_psnr'], 'mesh_psnr': step_out['mesh_psnr'], 'mesh_ssim': step_out['mesh_ssim']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'volumetric_psnr': step_out['volumetric_psnr'][oi], 'mesh_psnr': step_out['mesh_psnr'][oi], 'mesh_ssim': step_out['mesh_ssim'][oi]}
            volumetric_psnr = torch.mean(torch.stack([o['volumetric_psnr'] for o in out_set.values()]))
            mesh_psnr = torch.mean(torch.stack([o['mesh_psnr'] for o in out_set.values()]))
            mesh_ssim = torch.mean(torch.stack([o['mesh_ssim'] for o in out_set.values()]))
            self.log('test/volumetric_psnr', volumetric_psnr, prog_bar=True, rank_zero_only=True)
            self.log('test/mesh_psnr', mesh_psnr, prog_bar=True, rank_zero_only=True)
            self.log('test/mesh_ssim', mesh_ssim, prog_bar=True, rank_zero_only=True)

            psnr_path = os.path.join(self.save_dir, "test_psnr_results.txt")
            with open(psnr_path, "a") as f:
                f.write(
                    f"Step {self.global_step}: "
                    f"Volumetric PSNR = {volumetric_psnr.item():.4f}, "
                    f"Mesh PSNR = {mesh_psnr.item():.4f}\n"
                )
            print(f"[Saved PSNR] -> {psnr_path}")

            ssim_path = os.path.join(self.save_dir, "test_ssim_results.txt")
            with open(ssim_path, "a") as f:
                f.write(
                    f"Step {self.global_step}: "
                    f"Mesh SSIM = {mesh_ssim.item():.4f}\n"
                )
            print(f"[Saved SSIM] -> {ssim_path}")

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )

            print(f"[Saved Image Sequence] -> {os.path.join(self.save_dir, 'test.mp4')}")
            
            
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        


    def on_after_backward(self):
        point_grads = self.model.tetrahedral_grid.primal_points.grad
        if point_grads is not None:
            # Compute mean absolute gradient per point
            grad_mags = point_grads.abs().mean(dim=-1)

            # Create mask for finite values (no NaN, no Inf)
            finite_mask = torch.isfinite(grad_mags)

            # Perform EMA update only for finite entries
            ema = self.model.tetrahedral_grid.point_grad_accum
            ema[finite_mask] = ema[finite_mask] * 0.95 + (1 - 0.95) * grad_mags[finite_mask]

        # Optional: per-layer grad norms
        # for name, p in self.model.named_parameters():
        #     if p.grad is not None and p.grad.numel() > 0:
        #         # print(f"{name} abs grad mean: {p.grad.data.abs().mean():.4f}")
        #         # print(f"{name} abs grad max: {p.grad.data.abs().max():.4f}")
        #         #Print warning if gradient is inf or nan
        #         if not torch.isfinite(p.grad.data).all():
        #             print(f"Warning: {name} has non-finite gradients")
        #     else:
        #         pass
        #         print(f"{name} has no gradient")

    


# def MSE_loss_appearance(image, gt_image, model, view_idx, return_transformed_image=False):
#     appearance_embedding = model.get_appearance_embedding(view_idx)
#     # center crop the image
#     origH, origW = image.shape[1:-1]
#     H = origH // 32 * 32
#     W = origW // 32 * 32
#     left = origW // 2 - W // 2
#     top = origH // 2 - H // 2
#     crop_image = image[:, top:top+H, left:left+W]
#     crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
#     # down sample the image
#     crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
#     crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
#     mapping_image = model.appearance_embedding_network(crop_image_down)
#     transformed_image = mapping_image * crop_image
#     if not return_transformed_image:
#         return F.mse_loss(transformed_image, crop_gt_image)
#     else:
#         transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
#         return transformed_image



import torch
import torch.nn.functional as F

def MSE_loss_appearance(image, gt_image, model, view_idx, return_transformed_image=False, weights=None):
    """
    image, gt_image: (S, N, H, W, C) or (N, H, W, C)
    model.get_appearance_embedding(view_idx): (N, D)
    """
    has_s_dim = image.ndim == 5
    if has_s_dim:
        S, N_batch, origH, origW, C = image.shape
        appearance_embedding = model.get_appearance_embedding(view_idx.reshape(-1))
        image = image.reshape(S*N_batch, origH, origW, C)
        gt_image = gt_image.reshape(S*N_batch, origH, origW, C)
    else:
        origH, origW = image.shape[1:-1]
        appearance_embedding = model.get_appearance_embedding(view_idx)

    # move channels last -> first
    image = image.permute(0, 3, 1, 2)       # (B, C, H, W)
    gt_image = gt_image.permute(0, 3, 1, 2) # (B, C, H, W)

    # center crop
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, :, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, :, top:top+H, left:left+W]
    
    # downsample (preserve batch)
    crop_image_down = F.interpolate(crop_image, size=(H//32, W//32), mode="bilinear", align_corners=True)
    
    # expand appearance embedding spatially
    B, _, h_down, w_down = crop_image_down.shape
    D = appearance_embedding.shape[1]
    appearance_embedding_expanded = (
        appearance_embedding[:, :, None, None]
        .expand(-1, -1, h_down, w_down)
    )

    # concat and apply network
    crop_image_down = torch.cat([crop_image_down, appearance_embedding_expanded], dim=1)
    mapping_image = model.appearance_embedding_network(crop_image_down)
    transformed_image = mapping_image * crop_image

    if not return_transformed_image:
        if weights is not None and has_s_dim:
            w = weights.reshape(S*N_batch, origH, origW, 1).permute(0, 3, 1, 2)
            w = w[:, :, top:top+H, left:left+W]
            loss_unreduced = F.mse_loss(transformed_image, crop_gt_image, reduction='none')
            return (loss_unreduced * w).view(S, N_batch, -1).sum(0).mean()
        return F.mse_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = F.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)
        transformed_image = transformed_image.permute(0, 2, 3, 1)  # (B, H, W, C)
        if has_s_dim:
            transformed_image = transformed_image.reshape(S, N_batch, origH, origW, -1)
        return transformed_image
    

def L1_loss_appearance(image, gt_image, model, view_idx, return_transformed_image=False, weights=None):
    """
    image, gt_image: (S, N, H, W, C) or (N, H, W, C)
    model.get_appearance_embedding(view_idx): (N, D)
    """
    has_s_dim = image.ndim == 5
    if has_s_dim:
        S, N_batch, origH, origW, C = image.shape
        appearance_embedding = model.get_appearance_embedding(view_idx.reshape(-1))
        image = image.reshape(S*N_batch, origH, origW, C)
        gt_image = gt_image.reshape(S*N_batch, origH, origW, C)
    else:
        origH, origW = image.shape[1:-1]
        appearance_embedding = model.get_appearance_embedding(view_idx)

    # move channels last -> first
    image = image.permute(0, 3, 1, 2)       # (B, C, H, W)
    gt_image = gt_image.permute(0, 3, 1, 2) # (B, C, H, W)

    # center crop
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, :, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, :, top:top+H, left:left+W]
    
    # downsample (preserve batch)
    crop_image_down = F.interpolate(crop_image, size=(H//32, W//32), mode="bilinear", align_corners=True)
    
    # expand appearance embedding spatially
    B, _, h_down, w_down = crop_image_down.shape
    D = appearance_embedding.shape[1]
    appearance_embedding_expanded = (
        appearance_embedding[:, :, None, None]
        .expand(-1, -1, h_down, w_down)
    )

    # concat and apply network
    crop_image_down = torch.cat([crop_image_down, appearance_embedding_expanded], dim=1)
    mapping_image = model.appearance_embedding_network(crop_image_down)
    transformed_image = mapping_image * crop_image

    if not return_transformed_image:
        if weights is not None and has_s_dim:
            w = weights.reshape(S*N_batch, origH, origW, 1).permute(0, 3, 1, 2)
            w = w[:, :, top:top+H, left:left+W]
            loss_unreduced = F.l1_loss(transformed_image, crop_gt_image, reduction='none')
            return (loss_unreduced * w).view(S, N_batch, -1).sum(0).mean()
        return F.l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = F.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)
        transformed_image = transformed_image.permute(0, 2, 3, 1)  # (B, H, W, C)
        if has_s_dim:
            transformed_image = transformed_image.reshape(S, N_batch, origH, origW, -1)
        return transformed_image
