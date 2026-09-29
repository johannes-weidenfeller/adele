#!/usr/bin/env python
"""DTU Chamfer evaluation for a set of trained scans (paper protocol).

For every scan this (1) filters the raw output mesh by training-view visibility
(scripts/filter_dtu_meshes.py, vertex mode, n_min=2, 2x super-sampling) and
(2) runs the DTUeval-python protocol (scripts/eval_dtu/evaluate_single_scene.py)
against the official DTU point clouds. Per-scan results are written next to
the trial (`<exp_root>/radtets-scan<S>/<trial>/results_filtered/`) and
summarized in a CSV.

Usage:
    python scripts/eval_dtu_all.py --trial adele_dtu --gpu 0
    python scripts/eval_dtu_all.py --trial adele_dtu --scenes 24 37 --mask_cull
"""
import argparse, csv, os, re, subprocess, sys

SCENES = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', required=True, help='trial_name used at training time')
    ap.add_argument('--exp_root', default='./exp')
    ap.add_argument('--data_root', default='./load/dtu')
    ap.add_argument('--dtu_eval', default='./load/dtu_eval', help='official DTU eval data (ObsMask/, Points/stl/)')
    ap.add_argument('--scenes', nargs='+', type=int, default=SCENES)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--mask_cull', action='store_true',
                    help='additionally cull by the IDR 2D masks (NOT used for the paper numbers)')
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)

    scenes = ' '.join(str(s) for s in args.scenes)
    subprocess.run(f"{sys.executable} {ROOT}/scripts/filter_dtu_meshes.py --method {args.trial} "
                   f"--exp_root {args.exp_root} --data_root {args.data_root} --scenes {scenes} "
                   f"--mode vertex --n_min 2 --ss 2", shell=True, check=True, env=env)

    rows = []
    for s in args.scenes:
        trial_dir = os.path.join(args.exp_root, f'radtets-scan{s}', args.trial)
        mesh = os.path.join(trial_dir, 'save', 'final_mesh_filtered.ply')
        out = os.path.join(trial_dir, 'results_filtered')
        cull = '' if args.mask_cull else '--no_mask_cull'
        cmd = (f"{sys.executable} {ROOT}/scripts/eval_dtu/evaluate_single_scene.py --input_mesh {mesh} "
               f"--scan_id {s} --output_dir {out} --mask_dir {args.data_root} --DTU {args.dtu_eval} {cull}")
        print(f'[scan{s}] {cmd}', flush=True)
        res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
        print(res.stdout[-2000:])
        rf = os.path.join(out, 'results.json')
        overall = None
        if os.path.exists(rf):
            import json
            j = json.load(open(rf)); overall = j.get('overall')
        rows.append((s, overall))
        print(f'[scan{s}] chamfer={overall}')
    out_csv = args.out_csv or os.path.join(args.exp_root, f'dtu_{args.trial}_eval.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['scene', 'chamfer'])
        for s, o in rows: w.writerow([s, o])
        vals = [o for _, o in rows if o is not None]
        if vals: w.writerow(['mean', sum(vals) / len(vals)])
    print('wrote', out_csv)


if __name__ == '__main__':
    main()
