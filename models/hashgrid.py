import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from models.base import BaseModel
from models.utils import scale_anything, get_activation, cleanup, chunk_batch, contract_to_unisphere, get_contraction_type
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from utils.misc import get_rank
from systems.utils import update_module_step
from nerfacc import ContractionType, contract, contract_inv
import open3d as o3d
import matplotlib.pyplot as plt


#TODO: Implement per feature gradient scaling between MLP and encoding

@models.register('hashgrid')
class HashGrid(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        self.n_output_dims = self.config.feature_dim
        radius = self.config.radius
        print(f"[hashgrid.__init__] radius={radius} feature_dim={self.config.get('feature_dim', '?')} sphere_init_r={self.config.get('mlp_network_config', {}).get('sphere_init_radius', '?')}", flush=True)
        self.radius = torch.tensor(radius, dtype=torch.float32, device = get_rank())
        offset = self.config.get('offset', [0.0, 0.0, 0.0])
        self.offset = torch.tensor(offset, dtype=torch.float32, device = get_rank())
        self.contraction_type = get_contraction_type(self.config.get('contraction_type', 'aabb'))
        self.input_dim = self.config.get('input_dim', 3)
        if (self.input_dim == 3):
            self.roi = torch.tensor(
                        [
                            self.offset[0] - self.radius,
                            self.offset[1] - self.radius,
                            self.offset[2] - self.radius,
                            self.offset[0] + self.radius,
                            self.offset[1] + self.radius,
                            self.offset[2] + self.radius,
                        ],
                        dtype=torch.float32,
                        device=get_rank(),
            )
        else: 
            self.roi = torch.tensor(
                        [
                            self.offset[0] - self.radius,
                            self.offset[1] - self.radius,
                            self.offset[0] + self.radius,
                            self.offset[1] + self.radius,
                        ],
                        dtype=torch.float32,
                        device=get_rank(),
            )

        
        encoding = get_encoding(self.config.get('input_dim', 3), self.config.xyz_encoding_config)
        torch.manual_seed(42)
        np.random.seed(42)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network

        grad_scales = self.config.get('grad_scales', [1.0] * self.n_output_dims)
        self.register_buffer('grad_scales', torch.as_tensor(grad_scales, dtype=torch.float32))

    def forward(self, 
                points, 
                contract_input = True):
        
        if contract_input:
            roi_min = self.roi[:self.input_dim]
            roi_max = self.roi[self.input_dim:]
            if self.contraction_type == ContractionType.AABB:
                points = (points - roi_min) / (roi_max - roi_min)
            elif self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
                # MUST be fp32: volumetric ray-marching reaches far world radii
                # (far_plane), and (x-roi_min)/(roi_max-roi_min)*2-1 overflows fp16
                # (>65504) for r>~1e5 -> NaN -> illegal hashgrid index / CUDA crash.
                in_dtype = points.dtype
                points = points.float()
                points = (points - roi_min.float()) / (roi_max.float() - roi_min.float())
                points = points * 2.0 - 1.0
                norm = torch.norm(points, dim=-1, keepdim=True).clamp_min(1e-6)
                mask = norm > 1.0
                contraction_factor = (2.0*norm - 1.0) / (norm*norm)
                contracted = contraction_factor * points
                points = torch.where(mask, contracted, points)
                points = (points * 0.25 + 0.5).to(in_dtype)

            else:
                raise ValueError(f"Unknown contraction type: {self.contraction_type}")
                
        out = self.network(self.encoding(points.view(-1, self.input_dim))).view(*points.shape[:-1], self.n_output_dims).float()
        
        # Apply gradient scaling
        if self.grad_scales is not None:
             out = out * self.grad_scales + out.detach() * (1.0 - self.grad_scales)

        return out


    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.network, epoch, global_step)


@models.register('dual-hashgrid')
class DualHashGrid(BaseModel):
    """Two independent hashgrids for unbounded scenes, replacing the single
    contracted grid that made foreground and background share hash capacity:

      - fg grid: plain AABB hashgrid over the foreground box
        [-fg_radius, fg_radius]^3 -- full table capacity and finest resolution
        dedicated to the object region, no background collisions.
      - bg grid: the usual un_bounded_sphere contracted hashgrid over the whole
        domain (identical to the single-grid setup).

    Outputs transition at the foreground border with a radial smoothstep, so
    the SDF (and appearance features) stay C1-continuous:
        out(x) = (1-w(r)) * fg(x) + w(r) * bg(x),
        w = smoothstep over [fg_radius - blend_width, fg_radius],  r = |x-offset|.
    Inside the blend window both grids receive gradients (proportional to their
    weight), so they agree at the seam by construction.

    Sub-grids are standard HashGrid instances built from this config with
    optional `fg:`/`bg:` override blocks (deep-merged), so per-grid geometric
    MLP init applies: give fg `sphere_init_mode: single` (radius in fg-box
    passthrough units, i.e. r_world/fg_radius) and keep the double-sphere init
    on bg -- the blend then reproduces a consistent global double-sphere at t=0.

    forward() accepts the same conventions as HashGrid: model-space points with
    contract_input=True (all render paths), or nerfacc-contracted [0,1]^3 grid
    points with contract_input=False (grid-feature path), which are inverted
    back to model space here so both sub-grids and the blend see world radii.
    """

    def setup(self):
        from omegaconf import OmegaConf
        cfg = self.config
        self.n_output_dims = cfg.feature_dim
        self.radius = float(cfg.radius)
        self.fg_radius = float(cfg.get('fg_radius', self.radius))
        blend_width = float(cfg.get('blend_width', 0.2 * self.fg_radius))
        self.blend_lo = self.fg_radius - blend_width
        self.blend_hi = self.fg_radius
        offset = cfg.get('offset', [0.0, 0.0, 0.0])
        self.register_buffer('offset_t', torch.tensor(list(offset), dtype=torch.float32, device=get_rank()))

        base = OmegaConf.to_container(cfg, resolve=True)
        for k in ('name', 'fg', 'bg', 'fg_radius', 'blend_width'):
            base.pop(k, None)
        fg_cfg = OmegaConf.merge(OmegaConf.create(base), cfg.get('fg', {}) or {})
        bg_cfg = OmegaConf.merge(OmegaConf.create(base), cfg.get('bg', {}) or {})
        # fg grid: uncontracted AABB over the foreground box
        fg_cfg.radius = self.fg_radius
        fg_cfg.contraction_type = 'aabb'
        self.fg_grid = models.make('hashgrid', fg_cfg)
        self.bg_grid = models.make('hashgrid', bg_cfg)

    def forward(self, points, contract_input=True, **kwargs):
        if contract_input:
            xw = points.float()
        else:
            # invert the nerfacc [0,1]^3 un_bounded_sphere contraction the grid
            # stores its primal points in: [0,1] -> ball radius 2 -> undo
            # y = x*(2|x|-1)/|x|^2 (|y| = 2 - 1/|x|  =>  |x| = 1/(2-|y|))
            y = (points.float() - 0.5) * 4.0
            m = torch.norm(y, dim=-1, keepdim=True).clamp_min(1e-6)
            xn = torch.where(m > 1.0, y / (m * (2.0 - m).clamp_min(1e-3)), y)
            xw = xn * self.radius + self.offset_t

        r = torch.norm(xw - self.offset_t, dim=-1, keepdim=True)
        t = ((r - self.blend_lo) / (self.blend_hi - self.blend_lo)).clamp(0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)

        # fg queried on coords clamped into its box: out-of-box points have w=1
        # so the (garbage-free, clamped) fg value contributes exactly 0 there.
        fg_in = (xw - self.offset_t).clamp(-self.fg_radius, self.fg_radius) + self.offset_t
        fg_out = self.fg_grid(fg_in, contract_input=True)
        bg_out = self.bg_grid(xw, contract_input=True)
        return fg_out * (1.0 - w) + bg_out * w

    def update_step(self, epoch, global_step):
        update_module_step(self.fg_grid, epoch, global_step)
        update_module_step(self.bg_grid, epoch, global_step)