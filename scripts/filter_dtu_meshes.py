#!/usr/bin/env python3
"""Filter DTU output meshes by camera visibility.

For a given training method, walks the standard output layout
    <exp_root>/radtets-scan<S>/<method>/save/final_mesh.ply
loads the mesh and the COLMAP cameras the model was trained against, drops
any triangle that is *not visible* from at least `--n_min` cameras (default
2), and writes the filtered mesh next to the original as
    <exp_root>/radtets-scan<S>/<method>/save/final_mesh_filtered.ply

"Visible" here means: the triangle is the first hit (Z-buffer winner) of at
least one pixel in that camera's rasterization. So this is true rasterization
visibility — back faces and occluded surfaces are correctly excluded.

Camera handling exactly mirrors `datasets/colmap.py` (intrinsics from
cameras.bin, extrinsics from images.bin, COLMAP→OpenGL Y/Z column flip on
c2w, principal-point-corrected NDC). The saved mesh and these poses both
live in the original (un-normalized) DTU world frame, so no `normalize_poses`
adjustment is needed.

Usage:
    python scripts/filter_dtu_meshes.py --method dtu-simplified-high-batch
    python scripts/filter_dtu_meshes.py --method <m> --n_min 3 --exp_root ./exp
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

# The colmap reader lives in the datasets package — import it that way so we
# pick up the exact same parser the dataloader uses.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from datasets.colmap_utils import read_cameras_binary, read_images_binary  # noqa: E402


DEFAULT_DTU_SCENES = [37, 63, 69, 83, 105, 106, 110, 114, 118, 122, 97, 65, 40, 55, 24]


# ---------------------------------------------------------------------------
# Camera loading — kept structurally identical to datasets/colmap.py:174-248.
# ---------------------------------------------------------------------------

def _intrinsics_from_camdata(cam, factor=1.0):
    """Match the model→params dispatch in colmap.py:190-200 verbatim."""
    if cam.model == 'SIMPLE_RADIAL':
        fx = fy = cam.params[0] * factor
        cx = cam.params[1] * factor
        cy = cam.params[2] * factor
    elif cam.model in ('PINHOLE', 'OPENCV'):
        fx = cam.params[0] * factor
        fy = cam.params[1] * factor
        cx = cam.params[2] * factor
        cy = cam.params[3] * factor
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")
    return float(fx), float(fy), float(cx), float(cy)


def load_colmap_cameras(scene_dir):
    """Read COLMAP cameras the same way the training dataloader does.

    Returns:
        list of dicts with keys {fx, fy, cx, cy, W, H, c2w_opengl (4x4 np)}.
        c2w is already in the OpenGL convention (Y/Z columns flipped).
    """
    cameras_bin = os.path.join(scene_dir, 'sparse', '0', 'cameras.bin')
    images_bin = os.path.join(scene_dir, 'sparse', '0', 'images.bin')
    if not os.path.exists(cameras_bin) or not os.path.exists(images_bin):
        raise FileNotFoundError(
            f"Expected COLMAP files under {scene_dir}/sparse/0/. "
            f"Missing: {cameras_bin if not os.path.exists(cameras_bin) else images_bin}")

    camdata = read_cameras_binary(cameras_bin)
    imdata = read_images_binary(images_bin)

    # Single shared camera (camdata[1]) — matches colmap.py:176.
    cam = camdata[1]
    W, H = int(cam.width), int(cam.height)
    fx, fy, cx, cy = _intrinsics_from_camdata(cam, factor=1.0)  # full-res

    cams = []
    for d in imdata.values():
        R = d.qvec2rotmat()
        t = d.tvec.reshape(3, 1)
        # c2w in COLMAP convention: [R.T | -R.T @ t]  (colmap.py:246).
        c2w = np.concatenate([R.T, -R.T @ t], axis=1).astype(np.float64)
        c2w_4x4 = np.eye(4, dtype=np.float64)
        c2w_4x4[:3, :4] = c2w
        # COLMAP -> OpenGL: flip Y, Z columns of the 3x3 rotation block (colmap.py:247).
        c2w_4x4[:, 1:3] *= -1.0
        cams.append(dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H, c2w=c2w_4x4))
    return cams


# ---------------------------------------------------------------------------
# NDC matrix — mirrors colmap.py:215-220 (principal-point-corrected variant).
# ---------------------------------------------------------------------------

def make_ndc(fx, fy, cx, cy, W, H, near=0.01, far=1000.0):
    return np.array([
        [2.0 * fx / W, 0.0,            1.0 - 2.0 * cx / W,            0.0],
        [0.0,         -2.0 * fy / H,   1.0 - 2.0 * cy / H,            0.0],
        [0.0,          0.0,           -(far + near) / (far - near),  -(2 * far * near) / (far - near)],
        [0.0,          0.0,           -1.0,                           0.0],
    ], dtype=np.float64)


def make_mvp(cam, near=0.01, far=1000.0):
    ndc = make_ndc(cam['fx'], cam['fy'], cam['cx'], cam['cy'], cam['W'], cam['H'],
                   near=near, far=far)
    w2c = np.linalg.inv(cam['c2w'])
    return ndc @ w2c


# ---------------------------------------------------------------------------
# Core filter — Z-buffer visibility via nvdiffrast.
# ---------------------------------------------------------------------------

def _round_up_to_8(x):
    """nvdiffrast Cuda rasterizer requires resolution divisible by 8."""
    return ((x + 7) // 8) * 8


@torch.no_grad()
def filter_mesh_by_visibility(mesh, cameras, n_min=2, glctx=None,
                              device='cuda', verbose=True, ss=2,
                              mode='vertex'):
    """Filter a mesh by per-triangle rasterization visibility.

    Visibility is determined by rasterization: for each camera we run
    nvdiffrast and collect the set of triangle IDs that "won" at least one
    pixel. A triangle's visibility count is the number of cameras in which
    it appears in that set. A triangle is "visible" if count >= n_min.

    Two filtering modes:
        mode='face'   : drop every non-visible triangle. (Original behavior.)
        mode='vertex' : keep all triangles whose 3 vertices each belong to at
                        least one visible triangle. Equivalently: remove
                        vertices that are not connected to ANY visible
                        triangle, then drop faces that lose any vertex.
                        This preserves non-visible faces whose vertices are
                        anchored to the visible surface.

    Vertices that become unreferenced after face removal are dropped.

    Args:
        ss: super-sample factor. Rasterizes at `(ss*H, ss*W)` to catch tiny
            triangles that fall between pixel centers at native resolution
            (visible as gaps in the filtered mesh). The MVP is unchanged —
            same FOV / same NDC range, just denser pixel sampling. Memory
            cost grows ~ss^2 per camera; ss=2 is usually plenty for DTU.
    """
    if glctx is None:
        glctx = (dr.RasterizeGLContext() if os.environ.get("DTU_FILTER_GL") else dr.RasterizeCudaContext())
    assert ss >= 1 and int(ss) == ss, f"ss must be a positive integer, got {ss}"
    ss = int(ss)

    v = torch.tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
    f = torch.tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)
    n_tris = int(f.shape[0])
    counts = torch.zeros(n_tris, dtype=torch.int32, device=device)
    v_homog = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)  # (V, 4)

    # CudaRaster caps the viewport at 2048px; DTU at ss=2 is 3108x2324 which
    # crashes with CUDA error 700. Emulate ss=2 EXACTLY with ss^2 native-res
    # passes whose principal point is shifted so the union of their pixel
    # centers equals the supersampled grid's sample set.
    emulate = ss > 1 and (cameras[0]['H'] * ss > 2048 or cameras[0]['W'] * ss > 2048)
    if emulate:
        offs = [((k + 0.5) / ss - 0.5, (l + 0.5) / ss - 0.5)
                for k in range(ss) for l in range(ss)]
        print(f"    [ss{ss} emulation] viewport would exceed 2048px -> "
              f"{len(offs)} shifted native-res passes per camera")
    t0 = time.time()
    for ci, cam in enumerate(cameras):
        if emulate:
            seen = None
            for dx, dy in offs:
                c2 = dict(cam); c2['cx'] = cam['cx'] + dx; c2['cy'] = cam['cy'] + dy
                mvp = torch.tensor(make_mvp(c2), dtype=torch.float32, device=device)
                v_clip = (v_homog @ mvp.T).contiguous()[None]
                H_pad, W_pad = _round_up_to_8(cam['H']), _round_up_to_8(cam['W'])
                rast, _ = dr.rasterize(glctx, v_clip, f, resolution=[H_pad, W_pad])
                tri = rast[0, :cam['H'], :cam['W'], 3].long().reshape(-1)
                u = torch.unique(tri); u = u[u > 0] - 1
                seen = u if seen is None else torch.unique(torch.cat([seen, u]))
            counts[seen] += 1
            H_ss, W_ss = cam['H'] * ss, cam['W'] * ss  # for the progress line
        else:
            mvp = torch.tensor(make_mvp(cam), dtype=torch.float32, device=device)
            v_clip = (v_homog @ mvp.T).contiguous()[None]  # (1, V, 4)

            H_ss, W_ss = cam['H'] * ss, cam['W'] * ss
            H_pad, W_pad = _round_up_to_8(H_ss), _round_up_to_8(W_ss)
            rast, _ = dr.rasterize(glctx, v_clip, f, resolution=[H_pad, W_pad])
            # Crop back to the (super-sampled) image rect; padded rows/cols
            # never receive geometry but excluding them is just defensive.
            tri_id_1based = rast[0, :H_ss, :W_ss, 3].long().reshape(-1)
            unique = torch.unique(tri_id_1based)
            unique = unique[unique > 0] - 1  # 0 = no hit
            counts[unique] += 1
        if verbose and ((ci + 1) % 25 == 0 or ci + 1 == len(cameras)):
            print(f"      camera {ci + 1}/{len(cameras)} @ {W_ss}x{H_ss} "
                  f"(seen so far: {(counts > 0).sum().item()}/{n_tris} tris)")

    visible_tri_mask = (counts >= n_min)  # (n_tris,) bool on device

    if mode == 'face':
        keep_face_mask = visible_tri_mask.cpu().numpy()
    elif mode == 'vertex':
        # A vertex is "kept" iff it belongs to at least one visible triangle.
        n_verts = int(v.shape[0])
        vertex_kept = torch.zeros(n_verts, dtype=torch.bool, device=device)
        vis_face_vidx = f[visible_tri_mask].long().reshape(-1)
        vertex_kept[vis_face_vidx] = True
        # Keep faces whose all 3 vertices survive.
        f_long = f.long()
        keep_face_mask = (
            vertex_kept[f_long[:, 0]]
            & vertex_kept[f_long[:, 1]]
            & vertex_kept[f_long[:, 2]]
        ).cpu().numpy()
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'face' or 'vertex')")

    new_faces = mesh.faces[keep_face_mask]
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=new_faces,
        vertex_normals=mesh.vertex_normals if mesh.vertex_normals.shape[0] == mesh.vertices.shape[0] else None,
        process=False,  # don't merge/dedup vertices behind our back
    )
    # Now drop any vertex that's no longer referenced by any face.
    new_mesh.remove_unreferenced_vertices()

    if verbose:
        dt = time.time() - t0
        before, after = n_tris, int(keep_face_mask.sum())
        pct = 100.0 * (before - after) / max(before, 1)
        n_vis = int(visible_tri_mask.sum().item())
        print(f"      mode={mode}: visible tris={n_vis}/{before}, "
              f"kept {after}/{before} faces "
              f"(removed {pct:.1f}%) in {dt:.1f}s")
    return new_mesh


# ---------------------------------------------------------------------------
# Driver — iterate over scenes for a given method.
# ---------------------------------------------------------------------------

def _resolve_paths(args, scene):
    in_mesh = os.path.join(args.exp_root, f"radtets-scan{scene}", args.method,
                           "save", args.input_mesh)
    out_mesh = os.path.join(args.exp_root, f"radtets-scan{scene}", args.method,
                            "save", args.output_mesh)
    scene_dir = os.path.join(args.data_root, f"scan{scene}")
    return in_mesh, out_mesh, scene_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   help="Method / config name (matches `<exp_root>/radtets-scan<S>/<method>/...`).")
    p.add_argument("--exp_root", default="./exp",
                   help="Root of training outputs. Default: ./exp")
    p.add_argument("--data_root", default="./load/dtu",
                   help="Root of DTU scenes (each `scan<S>/sparse/0/...`). Default: ./load/dtu")
    p.add_argument("--scenes", nargs="+", type=int, default=DEFAULT_DTU_SCENES,
                   help="Scene IDs to process. Default: the standard 15-scene DTU set.")
    p.add_argument("--n_min", type=int, default=2,
                   help="Minimum number of cameras a triangle must be visible in to be kept.")
    p.add_argument("--view_ids", nargs="+", type=int, default=None,
                   help="Restrict visibility to these COLMAP-enumeration camera indices "
                        "(sparse-view runs: use only the TRAINING views, e.g. 12 11 2). "
                        "Default: all cameras.")
    p.add_argument("--mode", choices=['face', 'vertex'], default='vertex',
                   help="'face': drop non-visible triangles. "
                        "'vertex' (default): remove vertices unconnected to any "
                        "visible triangle, drop faces that lose a vertex.")
    p.add_argument("--ss", type=int, default=2,
                   help="Super-sample factor for rasterization (default 2). "
                        "Higher values catch smaller triangles at the cost of "
                        "~ss^2 memory and time per camera. Use 1 to disable.")
    p.add_argument("--input_mesh", default="final_mesh.ply",
                   help="Mesh filename inside <method>/save/. Default: final_mesh.ply")
    p.add_argument("--output_mesh", default="final_mesh_filtered.ply",
                   help="Output mesh filename in the same folder. Default: final_mesh_filtered.ply")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output meshes (default: skip if present).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    glctx = (dr.RasterizeGLContext() if os.environ.get("DTU_FILTER_GL") else dr.RasterizeCudaContext())

    n_done, n_skipped, n_failed = 0, 0, 0
    for scene in args.scenes:
        in_mesh, out_mesh, scene_dir = _resolve_paths(args, scene)
        print(f"\n[scan{scene}]")
        print(f"  mesh:    {in_mesh}")
        print(f"  cameras: {scene_dir}/sparse/0/")
        print(f"  output:  {out_mesh}")

        if not os.path.exists(in_mesh):
            print(f"  [skip] input mesh missing")
            n_skipped += 1
            continue
        if not os.path.isdir(scene_dir):
            print(f"  [skip] data dir missing")
            n_skipped += 1
            continue
        if os.path.exists(out_mesh) and not args.overwrite:
            print(f"  [skip] output already exists (pass --overwrite to redo)")
            n_skipped += 1
            continue

        try:
            mesh = trimesh.load(in_mesh, process=False)
            cams = load_colmap_cameras(scene_dir)
            if args.view_ids is not None:
                cams = [cams[i] for i in args.view_ids]
                print(f"  RESTRICTED to {len(cams)} training views {args.view_ids}")
            print(f"  loaded {len(mesh.faces)} faces, {len(cams)} cameras "
                  f"@ {cams[0]['W']}x{cams[0]['H']}")
            filtered = filter_mesh_by_visibility(
                mesh, cams, n_min=args.n_min, glctx=glctx,
                device=args.device, ss=args.ss, mode=args.mode)
            os.makedirs(os.path.dirname(out_mesh), exist_ok=True)
            filtered.export(out_mesh)
            print(f"  [done] -> {out_mesh}")
            n_done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {type(e).__name__}: {e}")
            n_failed += 1

    print(f"\nSummary: done={n_done}  skipped={n_skipped}  failed={n_failed}")


if __name__ == "__main__":
    main()
