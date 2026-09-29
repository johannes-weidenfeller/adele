#!/usr/bin/env python
"""Stanford ORB NVS evaluation of *extracted-mesh* renderings.

Renders the extracted vertex-colored mesh from every TEST view on both BLACK
and WHITE backgrounds for the methods we have available on ORB:
    ours             -> final_method_orb/save/final_mesh_baked_diffuse32.ply
    pgsr             -> PGSR_results/stanford_orb/<scene>/tsdf_fusion_post.ply
    pgsr_culled      -> PGSR_results/stanford_orb/<scene>/tsdf_fusion_post_culled.ply
    milo_learnable   -> MILO_results/stanford_orb/<scene>/mesh_learnable_sdf.ply
    milo_integration -> MILO_results/stanford_orb/<scene>/mesh_integration_sdf.ply
    (no tsdf_post variant exists for MILO on ORB.)

PSNR/SSIM use GT images composited over black via the test-image alpha channel.
ORB cameras come from transforms_test.json (NeRF-blender convention).

Outputs (per scene): same layout as nerf_synth_nvs.py.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
import nvdiffrast.torch as dr
from skimage.metrics import structural_similarity as ssim_fn

# Defaults; override with --exp_root/--trial/--data_root/--pgsr_root/--milo_root.
EXP_ROOT = './exp'
TRIAL = 'final_method_orb'
DATA_ROOT = './load/orb/blender_LDR'
PGSR_ROOT = None
MILO_ROOT = None
SCENES = ['ball_scene002', 'cactus_scene007', 'gnome_scene007',
          'pitcher_scene001', 'teapot_scene006']


def method_paths(scene: str) -> dict:
    """Mesh per method. 'ours' is the diffuse-baked mesh written by scripts/bake_vertex_colors.py;
    baseline meshes are only evaluated when --pgsr_root / --milo_root are given."""
    d = {'ours': os.path.join(EXP_ROOT, f'radtets-{scene}', TRIAL, 'save', 'final_mesh_baked_diffuse32.ply')}
    if PGSR_ROOT:
        d['pgsr'] = os.path.join(PGSR_ROOT, scene, 'tsdf_fusion_post.ply')
        d['pgsr_culled'] = os.path.join(PGSR_ROOT, scene, 'tsdf_fusion_post_culled.ply')
    if MILO_ROOT:
        d['milo_learnable'] = os.path.join(MILO_ROOT, scene, 'mesh_learnable_sdf.ply')
        d['milo_integration'] = os.path.join(MILO_ROOT, scene, 'mesh_integration_sdf.ply')
    return d


def load_blender_cams(scene_dir: str, split: str = 'test'):
    js = json.load(open(os.path.join(scene_dir, f'transforms_{split}.json')))
    fov_x = js['camera_angle_x']
    f0 = js['frames'][0]['file_path']
    img_path = os.path.join(scene_dir, f0 + '.png')
    if not os.path.exists(img_path):
        img_path = os.path.join(scene_dir, f0)
    W, H = Image.open(img_path).size
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    gl2cv = np.diag([1., -1., -1., 1.]).astype(np.float32)
    cams = []
    for fr in js['frames']:
        c2w_gl = np.asarray(fr['transform_matrix'], dtype=np.float32)
        c2w_cv = c2w_gl @ gl2cv
        w2c = np.linalg.inv(c2w_cv).astype(np.float32)
        stem = os.path.basename(fr['file_path'])
        cams.append({
            'name': stem,
            'gt_path':   os.path.join(scene_dir, fr['file_path'] + '.png'),
            'mask_path': os.path.join(scene_dir, f'{split}_mask', stem + '.png'),
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'W': W, 'H': H,
            'w2c': w2c,
        })
    return cams


def K_to_proj(fx, fy, cx, cy, W, H, n=0.01, f=100.0):
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 1 - 2 * cx / W
    P[1, 2] = 2 * cy / H - 1
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -2 * f * n / (f - n)
    P[3, 2] = -1
    return P


def load_mesh_gpu(mesh_path: str, device: str):
    m = trimesh.load(mesh_path, process=False)
    V = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32)).to(device)
    F = torch.from_numpy(np.asarray(m.faces, dtype=np.int32)).to(device)
    if m.visual.kind == 'vertex' and m.visual.vertex_colors is not None and m.visual.vertex_colors.shape[0] == V.shape[0]:
        C = torch.from_numpy(np.asarray(m.visual.vertex_colors[:, :3] / 255.0, dtype=np.float32)).to(device)
    else:
        C = torch.ones((V.shape[0], 3), device=device) * 0.7
    V_h = torch.cat([V, torch.ones_like(V[:, :1])], dim=-1)
    return V_h, F, C


def render(glctx, V_h, F, C, cam, bg=0.0):
    cv2gl = np.diag([1., -1., -1., 1.]).astype(np.float32)
    w2c_gl = cv2gl @ cam['w2c']
    proj = K_to_proj(cam['fx'], cam['fy'], cam['cx'], cam['cy'], cam['W'], cam['H'])
    mvp = torch.from_numpy(proj @ w2c_gl).cuda()
    clip = (V_h @ mvp.T)[None]
    rast, _ = dr.rasterize(glctx, clip, F, resolution=[cam['H'], cam['W']])
    col, _ = dr.interpolate(C[None], rast, F)
    col = col[0].clamp(0, 1).cpu().numpy()
    mask = (rast[0, ..., 3] > 0).cpu().numpy()
    out = np.full((cam['H'], cam['W'], 3), int(round(bg * 255)), dtype=np.uint8)
    out[mask] = (col[mask] * 255).astype(np.uint8)
    return out[::-1].copy()


def load_gt_over_black(gt_path: str, mask_path=None):
    """ORB test images are RGB on a black background but include the full
    scene (floor + supports). For a fair mesh-only comparison we mask GT
    using the test_mask (>0 = any object pixel) before computing PSNR.
    Without a mask, falls back to alpha compositing for NeRF-blender PNGs."""
    img = Image.open(gt_path)
    if img.mode == 'RGBA':
        arr = np.asarray(img).astype(np.float32) / 255.0
        rgb, a = arr[..., :3], arr[..., 3:]
        out = (rgb * a) * 255
    else:
        out = np.asarray(img.convert('RGB')).astype(np.float32)
    if mask_path and os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert('L'))
        out[m == 0] = 0
    return out.astype(np.uint8)


def psnr_u8(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12: return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_u8(a, b):
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def compose_compare(out_dir, methods_with_dirs, font):
    os.makedirs(out_dir, exist_ok=True)
    labels = [lab for lab, _ in methods_with_dirs]
    dirs   = [d   for _, d   in methods_with_dirs]
    if not dirs or not os.path.isdir(dirs[0]): return
    files = sorted(f for f in os.listdir(dirs[0]) if f.endswith('.png'))
    for fname in files:
        imgs = []
        for d in dirs:
            p = os.path.join(d, fname)
            if not os.path.exists(p): break
            imgs.append(np.asarray(Image.open(p).convert('RGB')))
        if len(imgs) != len(dirs): continue
        H = max(i.shape[0] for i in imgs)
        W_total = sum(i.shape[1] for i in imgs)
        header_h = 40
        canvas = Image.new('RGB', (W_total, H + header_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for img, lab in zip(imgs, labels):
            h, w = img.shape[:2]
            canvas.paste(Image.fromarray(img), (x, header_h))
            tw = draw.textlength(lab, font=font)
            draw.text((x + (w - tw) / 2, 6), lab, fill=(0, 0, 0), font=font)
            x += w
        canvas.save(os.path.join(out_dir, fname))


def process_scene(scene: str, args, glctx, font):
    paths = method_paths(scene)
    scene_dir = os.path.join(DATA_ROOT, scene)
    cams = load_blender_cams(scene_dir, split='test')
    print(f'[{scene}] {len(cams)} test views')

    out_root = os.path.join(EXP_ROOT, f'radtets-{scene}', TRIAL, 'save', 'nvs_eval')
    os.makedirs(out_root, exist_ok=True)

    avail = {m: p for m, p in paths.items() if os.path.exists(p)}
    if 'ours' not in avail:
        print(f'  [{scene}] SKIP: ours mesh missing'); return None
    if not avail: return None

    per_view = {m: {'psnr': [], 'ssim': []} for m in avail}
    gt_cache = {}

    for m, mesh_path in avail.items():
        t0 = time.time()
        V_h, F, C = load_mesh_gpu(mesh_path, 'cuda')
        dir_b = os.path.join(out_root, m, 'black'); os.makedirs(dir_b, exist_ok=True)
        dir_w = os.path.join(out_root, m, 'white'); os.makedirs(dir_w, exist_ok=True)
        for cam in cams:
            stem = cam['name']
            img_b = render(glctx, V_h, F, C, cam, bg=0.0)
            img_w = render(glctx, V_h, F, C, cam, bg=1.0)
            Image.fromarray(img_b).save(os.path.join(dir_b, f'{stem}.png'))
            Image.fromarray(img_w).save(os.path.join(dir_w, f'{stem}.png'))
            if stem not in gt_cache:
                gt_cache[stem] = load_gt_over_black(cam['gt_path'], cam.get('mask_path'))
            gt = gt_cache[stem]
            per_view[m]['psnr'].append(psnr_u8(img_b, gt))
            per_view[m]['ssim'].append(ssim_u8(img_b, gt))
        del V_h, F, C
        torch.cuda.empty_cache()
        print(f'  [{scene}] {m:18s} done in {time.time() - t0:.1f}s')

    comp_methods = [
        ('ours',           os.path.join(out_root, 'ours',           'white')),
        ('pgsr',           os.path.join(out_root, 'pgsr',           'white')),
        ('milo_learnable', os.path.join(out_root, 'milo_learnable', 'white')),
    ]
    comp_methods = [(lab, d) for lab, d in comp_methods if os.path.isdir(d)]
    if len(comp_methods) >= 2:
        compose_compare(os.path.join(out_root, 'compare_white'), comp_methods, font)
        print(f'  [{scene}] compare_white written')

    csv_path = os.path.join(out_root, 'nvs_psnr.csv')
    with open(csv_path, 'w') as f:
        f.write('method,view,psnr,ssim\n')
        for m, d in per_view.items():
            for stem, (p, s) in zip([c['name'] for c in cams], zip(d['psnr'], d['ssim'])):
                f.write(f'{m},{stem},{p:.4f},{s:.4f}\n')
    print(f'  [{scene}] wrote {csv_path}')

    return {m: {'psnr': float(np.mean(d['psnr'])),
                'ssim': float(np.mean(d['ssim']))}
            for m, d in per_view.items()}


def write_latex(table, scenes, methods, out_tex):
    with open(out_tex, 'w') as f:
        for metric in ['psnr', 'ssim']:
            f.write(f'%% --- {metric.upper()} ---\n')
            for m in methods:
                vals = []
                for s in scenes:
                    v = table.get(s, {}).get(m, {}).get(metric)
                    vals.append('—' if v is None else (f'{v:.3f}' if metric == 'ssim' else f'{v:.2f}'))
                nums = [float(x) for x in vals if x != '—']
                avg = (f'{sum(nums)/len(nums):.3f}' if metric == 'ssim' else f'{sum(nums)/len(nums):.2f}') if nums else '—'
                f.write(f'{m:18s} & ' + ' & '.join(vals) + f' & {avg} \\\\\n')
            f.write('\n')


def main():
    global EXP_ROOT, TRIAL, DATA_ROOT, PGSR_ROOT, MILO_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', default=SCENES)
    ap.add_argument('--gpu', default='4')
    ap.add_argument('--exp_root', default=EXP_ROOT, help='launch.py --exp_dir (default ./exp)')
    ap.add_argument('--trial', default=TRIAL, help='trial_name of the runs to evaluate')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--pgsr_root', default=None, help='optional dir with PGSR meshes per scene')
    ap.add_argument('--milo_root', default=None, help='optional dir with MILo meshes per scene')
    ap.add_argument('--out_summary_csv', default='nvs_summary.csv')
    ap.add_argument('--out_summary_tex', default='nvs_summary.tex')
    args = ap.parse_args()
    EXP_ROOT, TRIAL, DATA_ROOT = args.exp_root, args.trial, args.data_root
    PGSR_ROOT, MILO_ROOT = args.pgsr_root, args.milo_root
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f'GPU: {args.gpu}')

    glctx = dr.RasterizeCudaContext()
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
    except Exception:
        font = ImageFont.load_default()

    summary = {}
    methods_used = []
    for scene in args.scenes:
        try:
            res = process_scene(scene, args, glctx, font)
        except Exception as e:
            print(f'[{scene}] FAILED: {e}'); res = None
        if res is None: continue
        summary[scene] = res
        for m in res:
            if m not in methods_used: methods_used.append(m)

    rows = []
    for scene, res in summary.items():
        for m, d in res.items():
            rows.append((scene, m, d['psnr'], d['ssim']))
    with open(args.out_summary_csv, 'w') as f:
        f.write('scene,method,psnr,ssim\n')
        for r in rows: f.write(','.join(map(str, r)) + '\n')
    print(f'wrote {args.out_summary_csv}')

    method_order = ['pgsr', 'pgsr_culled', 'milo_learnable', 'milo_integration', 'ours']
    method_order = [m for m in method_order if m in methods_used]
    write_latex(summary, args.scenes, method_order, args.out_summary_tex)
    print(f'wrote {args.out_summary_tex}')

    print()
    print(f"{'scene':18s} " + ' '.join(f'{m:>20s}' for m in method_order))
    for s in args.scenes:
        if s not in summary: continue
        cells = []
        for m in method_order:
            d = summary[s].get(m)
            cells.append(f'{d["psnr"]:6.2f}/{d["ssim"]:.3f}' if d else '—')
        print(f'{s:18s} ' + ' '.join(f'{c:>20s}' for c in cells))


if __name__ == '__main__':
    main()
