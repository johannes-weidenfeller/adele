"""DTU-style visibility filtering for BMVS meshes.

Same rasterization-visibility filter as scripts/filter_dtu_meshes.py (drop
triangles not seen by >= n_min training cameras, ss-supersampled, vertex mode),
but reading BMVS/IDR cameras from <root_dir>/cameras.npz (world_mat_i @
scale_mat_i) instead of COLMAP sparse/0. The decomposed OpenCV pose is flipped
to the OpenGL convention the shared filter expects (c2w columns 1:3 negated),
matching how the training dataloader treats these cameras.

Usage:
  python scripts/filter_bmvs_mesh.py --root_dir load/bmvs/bear \
      --mesh exp/radtets-bear/<trial>/save/final_mesh.ply \
      --out  exp/radtets-bear/<trial>/save/final_mesh_filtered.ply \
      [--n_min 2] [--ss 2] [--mode vertex]
"""
import os, sys, argparse, glob
import numpy as np
import trimesh
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.filter_dtu_meshes import filter_mesh_by_visibility


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]
    K = K / K[2, 2]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, pose


def load_bmvs_cameras(root_dir, W, H):
    cams = []
    cd = np.load(os.path.join(root_dir, 'cameras.npz'))
    n = len([k for k in cd.files if k.startswith('world_mat_') and 'inv' not in k])
    for i in range(n):
        P = (cd[f'world_mat_{i}'] @ cd[f'scale_mat_{i}'])[:3, :4]
        K, c2w = load_K_Rt_from_P(P)
        c2w = c2w.copy()
        c2w[:, 1:3] *= -1.0                    # OpenCV -> OpenGL (repo convention)
        cams.append(dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                         cx=float(K[0, 2]), cy=float(K[1, 2]),
                         W=W, H=H, c2w=c2w.astype(np.float32)))
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', required=True)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_min', type=int, default=2)
    ap.add_argument('--ss', type=int, default=2)
    ap.add_argument('--mode', default='vertex', choices=['vertex', 'face'])
    ap.add_argument('--views', type=int, nargs='+', default=None,
                    help='restrict to these camera indices (e.g. the sparse-3view '
                         'TRAIN views) instead of all cameras in cameras.npz')
    a = ap.parse_args()

    img0 = sorted(glob.glob(os.path.join(a.root_dir, 'image', '*')))[0]
    import PIL.Image as Image
    W, H = Image.open(img0).size
    cams = load_bmvs_cameras(a.root_dir, W, H)
    if a.views is not None:
        cams = [cams[i] for i in a.views]
        print(f"[filter_bmvs] restricted to views {a.views}")
    mesh = trimesh.load(a.mesh, process=False)
    print(f"[filter_bmvs] {a.mesh}: {len(mesh.faces)} faces, {len(cams)} cams @ {W}x{H}")
    out = filter_mesh_by_visibility(mesh, cams, n_min=a.n_min, ss=a.ss, mode=a.mode)
    out.export(a.out)
    print(f"[filter_bmvs] kept {len(out.faces)}/{len(mesh.faces)} faces -> {a.out}")


if __name__ == '__main__':
    main()
