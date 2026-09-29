#!/usr/bin/env python
"""Bake per-vertex colors onto a saved mesh by querying the trained appearance
model head-on (view direction = -surface_normal).

Usage:
    python scripts/bake_vertex_colors.py \
        --run_dir exp/radtets-scan24/<trial> \
        --mesh    exp/radtets-scan24/<trial>/save/final_mesh_filtered_ss2.ply \
        --out     exp/radtets-scan24/<trial>/save/final_mesh_baked.ply
"""
import argparse, os, sys
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import datasets, systems  # noqa: register
from utils.misc import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=None, help='checkpoint .ckpt (defaults to ckpt/*.ckpt in run_dir)')
    ap.add_argument('--chunk', type=int, default=200_000)
    ap.add_argument('--diffuse_samples', type=int, default=1,
                    help='K hemisphere samples; >1 = cosine-weighted hemisphere average (diffuse).')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, 'config', 'parsed.yaml')
    config = OmegaConf.load(cfg_path)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpts = sorted([p for p in os.listdir(os.path.join(args.run_dir, 'ckpt')) if p.endswith('.ckpt')])
        assert ckpts, f'no ckpts under {args.run_dir}/ckpt'
        ckpt_path = os.path.join(args.run_dir, 'ckpt', ckpts[-1])
    print(f'[bake] cfg={cfg_path}\n[bake] ckpt={ckpt_path}')

    model = models.make(config.model.name, config.model).to(args.device).eval()
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    # Strip lightning's "model." prefix.
    sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    # Densification grew tet grid during training; resize params/buffers to match ckpt.
    msd = model.state_dict()
    for k, v in sd.items():
        if k in msd and msd[k].shape != v.shape:
            target = model
            *parents, leaf = k.split('.')
            for p in parents:
                target = getattr(target, p)
            cur = getattr(target, leaf)
            new = torch.nn.Parameter(v.clone().to(cur.device), requires_grad=cur.requires_grad) if isinstance(cur, torch.nn.Parameter) else v.clone().to(cur.device)
            if isinstance(cur, torch.nn.Parameter):
                setattr(target, leaf, new)
            else:
                target.register_buffer(leaf, new)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[bake] missing={len(missing)} unexpected={len(unexpected)}')

    mesh = trimesh.load(args.mesh, process=False)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    N = np.asarray(mesh.vertex_normals, dtype=np.float32)
    print(f'[bake] mesh: V={V.shape[0]} F={len(mesh.faces)}')

    # The saved mesh is in world space: world_v = (model_v * scale) @ transform_inv.T,
    # where transform_inv = inv(dataset.transform). To bake colors via the model we
    # must map back: model_v = (world_v @ dataset.transform.T) / scale.
    if not config.dataset.get('normalize_cameras', True):
        # e.g. BMVS final_method_bmvs: identity transform, scale=1.
        scale_f = 1.0
        T = torch.eye(4, device=args.device)
        print('[bake] normalize_cameras=false → using identity transform')
    else:
        from datasets.colmap import normalize_poses
        from datasets.colmap_utils import read_images_binary, read_points3d_binary
        root = config.dataset.root_dir
        imdata = read_images_binary(os.path.join(root, 'sparse/0/images.bin'))
        all_c2w = []
        for d in imdata.values():
            R = d.qvec2rotmat(); t = d.tvec.reshape(3, 1)
            c2w = torch.from_numpy(np.concatenate([R.T, -R.T @ t], axis=1)).float()
            c2w[:, 1:3] *= -1
            all_c2w.append(c2w)
        all_c2w = torch.stack(all_c2w, dim=0)
        pts3d_d = read_points3d_binary(os.path.join(root, 'sparse/0/points3D.bin'))
        pts3d = torch.from_numpy(np.array([pts3d_d[k].xyz for k in pts3d_d])).float()
        _, _, scale, transform = normalize_poses(
            all_c2w[:, :3, :], pts3d,
            up_est_method=config.dataset.up_est_method,
            center_est_method=config.dataset.center_est_method,
            scale_est_method=config.dataset.get('scale_est_method', 'min_cam'))
        scale_f = float(scale)
        T = transform.to(args.device).float()
        if T.shape == (3, 4):
            T = torch.cat([T, torch.tensor([[0., 0., 0., 1.]], device=T.device)], dim=0)
        print(f'[bake] scale={scale_f:.4f}  transform shape={tuple(T.shape)}')

    V_world_np = V.copy()
    V_world = torch.from_numpy(V).to(args.device)
    N_world = torch.from_numpy(N).to(args.device)
    V_h = torch.cat([V_world, torch.ones_like(V_world[:, :1])], dim=-1)
    V_model = (V_h @ T.T)[:, :3] / scale_f
    N_model = (T[:3, :3] @ N_world.T).T
    N_model = N_model / (N_model.norm(dim=-1, keepdim=True) + 1e-8)
    V = V_model.cpu().numpy()
    N = N_model.cpu().numpy()
    print(f'[bake] V_model range: min={V.min(0)} max={V.max(0)}')

    V_t = torch.from_numpy(V).to(args.device)
    N_t = torch.from_numpy(N).to(args.device)
    N_t = N_t / (N_t.norm(dim=-1, keepdim=True) + 1e-8)

    K = max(1, int(args.diffuse_samples))
    # Fibonacci hemisphere samples in canonical (z-up) frame.
    if K == 1:
        hemi = torch.tensor([[0., 0., 1.]], device=args.device)
    else:
        idx = torch.arange(K, device=args.device, dtype=torch.float32)
        # Cosine-weighted hemisphere (proportional to N·d): use 1 - (i+0.5)/K
        # for the squared-z component so density ∝ cos(theta).
        u = (idx + 0.5) / K
        z = torch.sqrt(1.0 - u)            # cosine-weighted: z = cos(theta) ∝ sqrt(1-u)
        r = torch.sqrt(u)
        ga = float(np.pi * (3.0 - np.sqrt(5.0)))  # golden angle
        phi = idx * ga
        hemi = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)
    print(f'[bake] diffuse_samples K={K}')

    def rotate_to_normal(hemi_dirs, n):
        # Build orthonormal frame around each normal n (B,3); return (B,K,3) world dirs.
        ref = torch.tensor([1., 0., 0.], device=n.device).expand_as(n).clone()
        mask = n[..., 0].abs() > 0.9
        ref[mask] = torch.tensor([0., 1., 0.], device=n.device)
        t = torch.cross(ref, n, dim=-1)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        b = torch.cross(n, t, dim=-1)
        R = torch.stack([t, b, n], dim=-1)  # (B,3,3) cols = [t,b,n]
        return torch.einsum('bij,kj->bki', R, hemi_dirs)  # (B,K,3)

    colors = torch.empty(V_t.shape[0], 3, device=args.device)
    with torch.no_grad():
        for i in range(0, V_t.shape[0], args.chunk):
            x = V_t[i:i + args.chunk]
            n = N_t[i:i + args.chunk]
            hg = model.get_hashgrid_features(x)
            sdf_in = model._collect_features(config.model.sdf.input_parameters, x, hg)
            sdf_out = model.sdf_model(**sdf_in)
            if isinstance(sdf_out, (tuple, list)):
                sdf_out = sdf_out[0]
            sdf = sdf_out[..., 0] if sdf_out.dim() == x.dim() else sdf_out

            if K == 1:
                d = -n
                app_in = model._collect_features(config.model.appearance.input_parameters, x, hg)
                rgb = model.appearance_model(dirs=d, normals=n, positions=x, sdf=sdf.unsqueeze(-1), **app_in)
                rgb = rgb.clamp(0.0, 1.0)
                colors[i:i + args.chunk] = rgb
                continue

            # K hemisphere directions per vertex (outward); camera sits along +d → pass -d as view dir.
            d_world = rotate_to_normal(hemi, n)                # (B,K,3) outward
            B = x.shape[0]
            xK = x[:, None, :].expand(B, K, 3).reshape(B * K, 3)
            nK = n[:, None, :].expand(B, K, 3).reshape(B * K, 3)
            sdfK = sdf[:, None].expand(B, K).reshape(B * K, 1)
            dirsK = (-d_world).reshape(B * K, 3)
            # Replicate hashgrid features instead of re-querying — same point.
            hgK = {k_: v[:, None, :].expand(B, K, v.shape[-1]).reshape(B * K, v.shape[-1])
                   for k_, v in hg.items()} if isinstance(hg, dict) else \
                   hg[:, None, :].expand(B, K, hg.shape[-1]).reshape(B * K, hg.shape[-1])
            app_in = model._collect_features(config.model.appearance.input_parameters, xK, hgK)
            rgb = model.appearance_model(dirs=dirsK, normals=nK, positions=xK, sdf=sdfK, **app_in)
            rgb = rgb.clamp(0.0, 1.0).reshape(B, K, 3)
            # Cosine-weighted average (importance sampling already cos-weighted -> uniform mean).
            colors[i:i + args.chunk] = rgb.mean(dim=1)

    colors_u8 = (colors.cpu().numpy() * 255).astype(np.uint8)
    rgba = np.concatenate([colors_u8, np.full((colors_u8.shape[0], 1), 255, dtype=np.uint8)], axis=-1)
    out = trimesh.Trimesh(vertices=V_world_np, faces=mesh.faces, vertex_colors=rgba, process=False)
    out.export(args.out)
    print(f'[bake] wrote {args.out}')


if __name__ == '__main__':
    main()
