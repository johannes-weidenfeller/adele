import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.utilities.rank_zero import rank_zero_info
import models
from models.base import BaseModel
from utils.misc import get_rank


@models.register('envmap')
class EnvironmentMap(BaseModel):
    """
    Learnable environment map represented as a cubemap texture.
    Maps a viewing direction (x, y, z) to an RGB color via direct lookup.

    Config options:
        resolution: int, resolution per cubemap face (default 256)
        n_channels: int, number of output channels (default 3)
        init_type: str, initialization type ['uniform', 'gaussian', 'zero']
        init_scale: float, scaling for init (default 0.1)
    """

    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        device = get_rank()
        self.resolution = int(self.config.get('resolution', 256))
        self.n_channels = int(self.config.get('n_channels', 3))
        self.init_type = self.config.get('init_type', 'fixed_color')
        self.init_scale = float(self.config.get('init_scale', 0.1))
        self.init_color = self.config.get('init_color', [0.5, 0.5, 0.5])

        # 6 cube faces: +X, -X, +Y, -Y, +Z, -Z
        self.cubemap = nn.Parameter(
            torch.empty(6, self.n_channels, self.resolution, self.resolution, device=device)
        )

        # Initialize
        if self.init_type == 'uniform':
            nn.init.uniform_(self.cubemap, -self.init_scale, self.init_scale)
        elif self.init_type == 'gaussian':
            nn.init.normal_(self.cubemap, mean=0.0, std=self.init_scale)
        elif self.init_type == 'zero':
            nn.init.zeros_(self.cubemap)
        elif self.init_type == 'fixed_color':
            color = torch.tensor(self.init_color, dtype=torch.float32, device=device)
            assert color.numel() == self.n_channels, \
                f"init_color must have {self.n_channels} values (got {color.numel()})"
            self.cubemap.data[:] = color.view(1, self.n_channels, 1, 1)
        else:
            raise ValueError(f"Unknown init_type={self.init_type}")

        rank_zero_info(f"Initialized learned environment map with resolution={self.resolution} and n_channels={self.n_channels}")

    def direction_to_face_uv(self, dirs):
        """
        Convert normalized direction vectors (x, y, z) into
        cubemap face indices and UV coordinates.

        dirs: (N, 3) normalized direction vectors
        Returns: face_idx (N,), uv (N, 2) in [0,1]^2
        """
        abs_dirs = dirs.abs()
        major_axis = abs_dirs.argmax(dim=-1)  # (N,)
        face_idx = torch.zeros_like(major_axis)

        # Allocate output UV
        uv = torch.zeros_like(dirs[..., :2])  # (N, 2)

        # Mapping for each major axis
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        # +X face
        mask = (major_axis == 0) & (x > 0)
        face_idx[mask] = 0
        uv[mask, 0] = -z[mask] / x[mask]
        uv[mask, 1] = -y[mask] / x[mask]

        # -X face
        mask = (major_axis == 0) & (x <= 0)
        face_idx[mask] = 1
        uv[mask, 0] = z[mask] / -x[mask]
        uv[mask, 1] = -y[mask] / -x[mask]

        # +Y face
        mask = (major_axis == 1) & (y > 0)
        face_idx[mask] = 2
        uv[mask, 0] = x[mask] / y[mask]
        uv[mask, 1] = z[mask] / y[mask]

        # -Y face
        mask = (major_axis == 1) & (y <= 0)
        face_idx[mask] = 3
        uv[mask, 0] = x[mask] / -y[mask]
        uv[mask, 1] = -z[mask] / -y[mask]

        # +Z face
        mask = (major_axis == 2) & (z > 0)
        face_idx[mask] = 4
        uv[mask, 0] = x[mask] / z[mask]
        uv[mask, 1] = -y[mask] / z[mask]

        # -Z face
        mask = (major_axis == 2) & (z <= 0)
        face_idx[mask] = 5
        uv[mask, 0] = -x[mask] / -z[mask]
        uv[mask, 1] = -y[mask] / -z[mask]

        # Map [-1,1] → [0,1]
        uv = (uv + 1.0) * 0.5
        uv = uv.clamp(0.0, 1.0)

        return face_idx, uv

    def sample_cubemap(self, dirs):
        """
        Efficient cubemap sampling for millions of directions.
        Uses fully vectorized bilinear interpolation (no per-face loops).

        dirs: (N, 3) normalized direction vectors
        Returns: (N, n_channels)
        """
        face_idx, uv = self.direction_to_face_uv(dirs)  # (N,), (N, 2)
        N = dirs.shape[0]
        H = W = self.resolution
        C = self.n_channels

        # Convert [0,1] UV -> pixel coordinates
        u = uv[:, 0] * (W - 1)
        v = uv[:, 1] * (H - 1)

        # Integer pixel coordinates (clamped)
        u0 = u.floor().long().clamp(0, W - 1)
        v0 = v.floor().long().clamp(0, H - 1)
        u1 = (u0 + 1).clamp(0, W - 1)
        v1 = (v0 + 1).clamp(0, H - 1)

        # Bilinear interpolation weights
        wu = u - u0.float()
        wv = v - v0.float()
        w00 = (1 - wu) * (1 - wv)
        w10 = wu * (1 - wv)
        w01 = (1 - wu) * wv
        w11 = wu * wv

        tex = self.cubemap  # (6, C, H, W)

        # Gather pixel colors for all samples (N, C)
        def gather(face, uu, vv):
            return tex[face, :, vv, uu]  # (N, C)

        c00 = gather(face_idx, u0, v0)
        c10 = gather(face_idx, u1, v0)
        c01 = gather(face_idx, u0, v1)
        c11 = gather(face_idx, u1, v1)

        # Weighted blend
        out = (
            w00[:, None] * c00 +
            w10[:, None] * c10 +
            w01[:, None] * c01 +
            w11[:, None] * c11
        )

        return out

    def forward(self, dirs, normalize=False):
        """
        dirs: (..., 3) viewing directions
        Returns: (..., n_channels) background color
        """
        orig_shape = dirs.shape[:-1]  # save original leading shape

        dirs = dirs.view(-1, 3)
        if normalize:
            dirs = F.normalize(dirs, dim=-1)

        color = self.sample_cubemap(dirs)
        return color.view(*orig_shape, self.n_channels)

    def total_variation_loss(self):
        """Simple TV regularizer to encourage smoothness."""
        loss = 0
        for f in range(6):
            img = self.cubemap[f]
            loss += (
                torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])) +
                torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
            )
        return loss / 6.0

    def update_step(self, epoch, global_step):
        # No per-step update needed, but keep interface consistent
        pass