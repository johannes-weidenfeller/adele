"""Background sphere: a large, fixed-geometry sphere enclosing the scene whose
appearance is stored per-vertex and looked up by ray direction. Only the vertex
appearance is optimised — the geometry (vertex directions) is frozen. This gives
a cheap, well-behaved background (a mesh-based environment map) for unbounded
scenes, replacing the learned volumetric background.

For a ray direction d (unit), the background colour is a direction-weighted blend
of the K nearest sphere vertices' colours — i.e. per-vertex appearance smoothly
interpolated over the sphere. No ray-tracing / OptiX involved.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models


def _fibonacci_sphere(n):
    """n roughly-uniform unit vectors on the sphere (Fibonacci lattice)."""
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5 ** 0.5) / 2
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    theta = 2 * math.pi * i / phi
    return np.stack([r * np.cos(theta), z, r * np.sin(theta)], -1)  # (n,3)


@models.register('sphere-bg')
class BackgroundSphere(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n = int(config.get('num_vertices', 2000))
        self.topk = int(config.get('topk', 16))
        self.temperature = float(config.get('temperature', 0.02))
        # frozen geometry: unit directions of the sphere vertices
        verts = torch.from_numpy(_fibonacci_sphere(n)).float()
        self.register_buffer('verts', F.normalize(verts, dim=-1))   # (V,3), NOT a Parameter
        # learnable per-vertex appearance (pre-sigmoid RGB logits), init to mid-grey
        self.colors = nn.Parameter(torch.zeros(n, 3))

    def forward(self, dirs):
        """dirs: (..., 3) ray directions -> (..., 3) background colour in [0,1]."""
        lead = dirs.shape[:-1]
        d = F.normalize(dirs.reshape(-1, 3), dim=-1)                 # (P,3)
        sim = d @ self.verts.t()                                     # (P,V) cosine similarity
        topv, topi = sim.topk(self.topk, dim=-1)                     # (P,K)
        w = torch.softmax(topv / self.temperature, dim=-1)          # (P,K) direction weights
        cols = torch.sigmoid(self.colors)[topi]                      # (P,K,3)
        bg = (w.unsqueeze(-1) * cols).sum(dim=1)                     # (P,3)
        return bg.reshape(*lead, 3)
