"""Evaluate BMVS reconstructions of a method against ground-truth point clouds.

For each scene `<S>` in --scenes, this script:
    pred = {exp_root}/radtets-<S>/{method_name}/save/final_mesh.ply
    gt   = {dataset_path}/<S>/gt_pts.ply
and computes the symmetric Chamfer distance after triangle resampling, scaled
to mm via scale=1e3 (matching the evaluation convention of the provided code).

Adapted from https://github.com/jzhangbs/DTUeval-python via the BMVS-flavored
`eval_simple` snippet shared by the user. Wrapper structure mirrors
`evaluate_orb.py` so per-scene results land in a CSV alongside an AVERAGE row.

Usage:
    python scripts/eval_bmvs.py --method_name <trial_name>
    # or override anything:
    python scripts/eval_bmvs.py \
        --method_name <trial> \
        --dataset_path ./load/bmvs \
        --exp_root ./exp \
        --scenes bear cow stone \
        --output_filename bmvs_my_method.txt
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm


# Default scene list (mirrors scripts/run_bmvs.py).
DEFAULT_SCENES = [
    'bear', 'bread', 'camera', 'clock', 'cow', 'dog', 'doll', 'dragon',
    'durian', 'fountain', 'gundam', 'house', 'jade', 'man', 'monster',
    'sculpture', 'stone', 'basketball',
]


# ---------------------------------------------------------------------------
# Core eval (verbatim from the user-provided snippet, with light hardening).
# ---------------------------------------------------------------------------
def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:int(n1) + 1, :int(n2) + 1].astype(np.float64)
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    """Save a coloured point cloud (mirrors scripts/eval_dtu/eval.py)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def eval_simple(in_file, stl_file, scale, vis_out_dir=None, vis_basename=None,
                vis_threshold=None):
    """Symmetric chamfer between a mesh (in_file) and a point cloud (stl_file).

    `scale` rescales the input distances into the units the BMVS GT lives in
    (here 1e3 = millimetre convention). The returned scalar has those units.

    If `vis_out_dir` is provided, also writes two coloured PLYs there:
        {vis_basename}_d2s.ply : the mesh-sampled point cloud, coloured by
                                 distance to the GT cloud (white = on-surface,
                                 red = up to vis_threshold, green = clipped
                                 outlier beyond max_dist).
        {vis_basename}_s2d.ply : the GT cloud, coloured by distance to the
                                 mesh-sampled cloud (same colour scheme).
    `vis_threshold` is in the SAME units as the returned chamfer (e.g. mm
    when scale=1e3); defaults to max_dist*scale (i.e. 20 mm).
    """
    data_mesh = o3d.io.read_triangle_mesh(str(in_file))
    if len(data_mesh.vertices) == 0 or len(data_mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no vertices/triangles: {in_file}")
    data_mesh.remove_unreferenced_vertices()

    mp.freeze_support()

    # default DTU values (also used in the original BMVS adaptation)
    max_dist = 20 / scale
    # 0.2 is the DTU convention in mm; convert to scene units like max_dist.
    # (previously left unscaled -> prediction side was sampled ~1000x too
    # coarsely, i.e. effectively just the mesh vertices)
    thresh = 0.2 / scale  # supersampling density

    pbar = tqdm(total=4, leave=False)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)
    # hard sample budget: tiny-triangle meshes + fine thresh explode the
    # subdivision (100s of millions of points -> eval hangs). If the estimate
    # exceeds the budget, relax the spacing uniformly (sqrt scaling) so counts
    # land at ~MAX_SAMPLES while keeping density uniform across the surface.
    MAX_SAMPLES = 5_000_000
    est = float((np.maximum(n1, 1) * np.maximum(n2, 1) / 2).sum())
    if est > MAX_SAMPLES:
        relax = np.sqrt(est / MAX_SAMPLES)
        thr = thr * relax
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)
        print(f"[eval_bmvs] sample budget: est {est/1e6:.0f}M -> relaxing spacing "
              f"x{relax:.2f} (effective ~{thresh*relax*1e3:.2f}mm-equiv)")

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(
            sample_single_tri,
            ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0])
             for i in range(len(n1))),
            chunksize=1024,
        )

    new_pts = np.concatenate(new_pts, axis=0) if len(new_pts) else np.zeros((0, 3))
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(stl_file)
    stl = np.asarray(stl_pcd.points)
    if stl.size == 0:
        pbar.close()
        raise RuntimeError(f"GT point cloud is empty: {stl_file}")

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, _ = nn_engine.kneighbors(data_pcd, n_neighbors=1, return_distance=True)
    valid_d2s = dist_d2s[dist_d2s < max_dist]
    mean_d2s = valid_d2s.mean() if valid_d2s.size else float('nan')

    pbar.update(1)
    pbar.set_description('compute stl2data')
    nn_engine.fit(data_pcd)
    dist_s2d, _ = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    valid_s2d = dist_s2d[dist_s2d < max_dist]
    mean_s2d = valid_s2d.mean() if valid_s2d.size else float('nan')

    # ---- Optional: write coloured per-point error PLYs (mirrors eval_dtu) ----
    if vis_out_dir is not None:
        os.makedirs(vis_out_dir, exist_ok=True)
        # `vis_threshold` is the user-facing knob in reported-chamfer units (mm
        # for BMVS at scale=1e3); convert to internal distance units.
        if vis_threshold is None:
            vis_dist = max_dist
        else:
            vis_dist = float(vis_threshold) / scale

        R = np.array([[1, 0, 0]], dtype=np.float64)
        G = np.array([[0, 1, 0]], dtype=np.float64)
        W = np.array([[1, 1, 1]], dtype=np.float64)

        # data2stl: colour every mesh-sample point. Outliers (>= max_dist)
        # render green; everything else lerps white -> red as distance grows.
        data_alpha = np.clip(dist_d2s, 0.0, vis_dist) / max(vis_dist, 1e-12)
        data_color = R * data_alpha + W * (1.0 - data_alpha)
        data_color[dist_d2s[:, 0] >= max_dist] = G
        write_vis_pcd(os.path.join(vis_out_dir, f"{vis_basename}_d2s.ply"),
                      data_pcd, data_color)

        # stl2data: same colouring but on the GT point cloud.
        stl_alpha = np.clip(dist_s2d, 0.0, vis_dist) / max(vis_dist, 1e-12)
        stl_color = R * stl_alpha + W * (1.0 - stl_alpha)
        stl_color[dist_s2d[:, 0] >= max_dist] = G
        write_vis_pcd(os.path.join(vis_out_dir, f"{vis_basename}_s2d.ply"),
                      stl, stl_color)

    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2 * scale
    # stash components (mm) so callers can report accuracy (d2s) and
    # completeness (s2d) separately -- the CSV wrapper only kept the mean
    eval_simple.last_components = (mean_d2s * scale, mean_s2d * scale)
    return over_all


def eval_bmvs(source_path, in_mesh, vis_out_dir=None, vis_basename=None,
              vis_threshold=None, gt_filename='gt_pts.ply'):
    """BMVS-style chamfer: symmetric distance to gt_pts.ply, returned in mm.

    See `eval_simple` for the visualisation arguments.
    """
    gt_path = os.path.join(source_path, gt_filename)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT point cloud missing: {gt_path}")
    return eval_simple(in_mesh, gt_path, scale=1e3,
                       vis_out_dir=vis_out_dir, vis_basename=vis_basename,
                       vis_threshold=vis_threshold)


# ---------------------------------------------------------------------------
# Wrapper: per-scene loop, CSV output (mirrors evaluate_orb.py).
# ---------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="Evaluate BMVS reconstructions of a method.")
    p.add_argument("--dataset_path", type=str, default="./load/bmvs",
                   help="Root containing per-scene subdirectories with cameras.npz, gt_pts.ply.")
    p.add_argument("--exp_root", type=str, default="./exp",
                   help="Root of training outputs (.../radtets-<scene>/<method>/save/final_mesh.ply).")
    p.add_argument("--method_name", type=str, required=True,
                   help="Trial name under exp_root/radtets-<scene>/. Same value as launch.py's trial_name.")
    p.add_argument("--output_filename", type=str, default=None,
                   help="Output CSV path. Defaults to bmvs_<method_name>.txt in CWD.")
    p.add_argument("--scenes", type=str, nargs="+", default=DEFAULT_SCENES,
                   help="Subset of scenes to evaluate.")
    p.add_argument("--gt_filename", type=str, default="gt_pts.ply",
                   help="GT point cloud filename per scene dir (e.g. sparse_gt.ply "
                        "for the train-view-visibility-filtered GT).")
    p.add_argument("--mesh_filename", type=str, default="final_mesh.ply",
                   help="Filename of the predicted mesh under <method>/save/.")
    p.add_argument("--no_vis", action="store_true",
                   help="Skip writing per-point error PLYs (vis_*_d2s.ply, vis_*_s2d.ply).")
    p.add_argument("--vis_threshold", type=float, default=10.0,
                   help="Distance (in reported-chamfer units, i.e. mm) at which the "
                        "red->white colour ramp saturates. Outliers >= max_dist are green.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    output_filename = args.output_filename or f"bmvs_{args.method_name}.txt"

    print(f"Method:        {args.method_name}")
    print(f"Dataset path:  {args.dataset_path}")
    print(f"Exp root:      {args.exp_root}")
    print(f"Scenes ({len(args.scenes)}): {args.scenes}")
    print(f"Output:        {output_filename}\n")

    chamfer_results = []
    with open(output_filename, "w") as f:
        f.write(f"# BMVS chamfer evaluation (scale=1e3 -> distances reported in mm)\n")
        f.write(f"# method={args.method_name}  exp_root={args.exp_root}  dataset_path={args.dataset_path}\n")
        f.write("Scene Name, Chamfer (mm)\n")
        f.write("-" * 60 + "\n")
        f.flush()

        for scene in args.scenes:
            source_path = os.path.join(args.dataset_path, scene)
            pred_path = os.path.join(args.exp_root, f"radtets-{scene}",
                                     args.method_name, "save", args.mesh_filename)

            print(f"=== {scene} ===")
            if not os.path.isdir(source_path):
                print(f"  [Error] Scene dir missing: {source_path}")
                f.write(f"{scene}, ERROR_SCENE_MISSING\n"); f.flush()
                continue
            if not os.path.exists(pred_path):
                print(f"  [Error] Prediction missing: {pred_path}")
                f.write(f"{scene}, ERROR_PRED_MISSING\n"); f.flush()
                continue

            print(f"  pred: {pred_path}")
            print(f"  gt:   {os.path.join(source_path, 'gt_pts.ply')}")

            if args.no_vis:
                vis_out_dir = None
                vis_basename = None
            else:
                # Sit alongside the trial's `save/` folder, mirroring DTU's layout.
                vis_out_dir = os.path.join(args.exp_root, f"radtets-{scene}",
                                           args.method_name, "results")
                vis_basename = f"vis_{scene}"

            try:
                cd = eval_bmvs(source_path, pred_path,
                               vis_out_dir=vis_out_dir,
                               vis_basename=vis_basename,
                               vis_threshold=args.vis_threshold,
                               gt_filename=args.gt_filename)
            except Exception as e:
                print(f"  [Error] Eval failed: {e}")
                f.write(f"{scene}, ERROR_EVAL ({e})\n"); f.flush()
                continue

            d2s, s2d = getattr(eval_simple, 'last_components', (float('nan'),) * 2)
            print(f"  CD = {cd:.6f} mm  (d2s/acc = {d2s:.6f}, s2d/comp = {s2d:.6f})")
            if vis_out_dir is not None:
                print(f"  vis: {vis_out_dir}/{vis_basename}_{{d2s,s2d}}.ply")
            f.write(f"{scene}, {cd:.6f}, d2s={d2s:.6f}, s2d={s2d:.6f}\n"); f.flush()
            chamfer_results.append(cd)

        f.write("-" * 60 + "\n")
        if chamfer_results:
            avg = float(np.mean(chamfer_results))
            print("=" * 40)
            print(f"Average CD over {len(chamfer_results)}/{len(args.scenes)} scenes: {avg:.6f} mm")
            f.write(f"AVERAGE ({len(chamfer_results)}/{len(args.scenes)}), {avg:.6f}\n")
        else:
            print("No valid results.")
            f.write("AVERAGE, N/A\n")

    print(f"\nDone. Results saved to {output_filename}")


if __name__ == "__main__":
    main()
