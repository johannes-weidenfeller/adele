# ADELE

Official implementation of **"ADELE - Adaptive Delaunay Grids for High-Fidelity Mesh-Native Reconstruction"** (SIGGRAPH Asia 2026 Conference Papers).

[Johannes Weidenfeller](https://vlg.inf.ethz.ch/team/Johannes-Weidenfeller.html), [Shaofei Wang](https://taconite.github.io), [Philipp Fürnstahl](https://health.dsi.uzh.ch/member/philipp-furnsthal/), [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html) · ETH Zurich

[Project page](https://johannes-weidenfeller.github.io/adele) · [arXiv](https://arxiv.org/abs/XXXXX)

ADELE reconstructs watertight, detailed meshes from multi-view images by directly
optimizing an adaptive Delaunay tetrahedral grid together with a multi-resolution
hash-grid SDF. Training is hybrid: a NeuS-style volumetric branch bootstraps the
coarse geometry, and a differentiable mesh rasterizer (nvdiffrast) with
depth-offset sampling recovers fine detail. The grid is refined by gradient-driven
vertex insertion and pruning with periodic Delaunay re-triangulation (RadFoam).

This repository contains the training code, the configurations used for every
table in the paper, and the evaluation scripts for DTU, BlendedMVS, NeRF-Synthetic
and Stanford-ORB.

## Contents

- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Memory and runtime](#memory-and-runtime)
- [Repository layout](#repository-layout)
- [License](#license)
- [Citation](#citation)

## Installation

The code needs Linux, an NVIDIA GPU with compute capability 8.0 or newer
(tested: RTX 4090 / sm_89 and H200 / sm_90), a CUDA 12.x toolkit with `nvcc`,
and Python 3.10. Three dependencies compile CUDA code at install time
(tiny-cuda-nn, RadFoam) or on first import (nerfacc, nvdiffrast, the bundled
`render/renderutils` plugin), so a working `nvcc` matching your PyTorch build is
essential.

The recipe below was verified from scratch (conda, no system CUDA needed; the CUDA
12.8 toolkit and a pinned GCC come from conda). Budget about one hour on a desktop,
most of it compiling tiny-cuda-nn and RadFoam.

```bash
# 1. environment: Python 3.10, CUDA 12.8 toolkit, GCC 11.2 (GCC 14, which cuda-toolkit
#    pulls in by default, breaks the nerfacc and RadFoam builds)
conda create -y -n adele python=3.10
conda install -y -n adele -c nvidia/label/cuda-12.8.1 cuda-toolkit
conda install -y -n adele "gcc_linux-64=11.2" "gxx_linux-64=11.2"
conda activate adele
source env.sh            # CUDA_HOME, include/lib paths, JIT cache dir; source it before every run too

# 2. PyTorch (cu128 wheels must match the conda toolkit version)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 3. python dependencies
pip install -r requirements.txt

# 4. nvdiffrast (v0.3.3, JIT-compiled on first use)
pip install git+https://github.com/NVlabs/nvdiffrast@v0.3.3

# 5. tiny-cuda-nn (compiles one extension per listed architecture; keep only yours to save time,
#    e.g. TCNN_CUDA_ARCHITECTURES=89 for an RTX 4090)
TCNN_CUDA_ARCHITECTURES="80;86;89;90" pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn@dc44d0d#subdirectory=bindings/torch"

# 6. RadFoam (GPU Delaunay triangulation); --recursive is required, the clone is ~1.5 GB
git clone --recursive https://github.com/theialab/radfoam.git
cd radfoam && git checkout 3e7b52cf74e37ab2ab5e695f53570f515f537e3d
CUDAARCHS="80;86;89;90" pip install .
cd ..

# 7. build the JIT extensions once (nerfacc ~2 min, nvdiffrast + renderutils < 1 min)
python -c "from nerfacc.cuda._backend import _C"
python -c "import sys; sys.path.insert(0,'.'); from render.renderutils import ops; ops._get_plugin(); import nvdiffrast.torch as dr; dr.RasterizeCudaContext()"
```

Notes:

- `env.sh` must be sourced in every shell that trains or evaluates: the JIT extensions need the CUDA include/library paths at runtime, and `TORCH_EXTENSIONS_DIR` keeps this environment's build cache separate from other environments.
- RadFoam's Python package loads GLFW at import and needs the X11 runtime libraries (`libX11.so.6`); they are present on any desktop, on a bare server install `libx11-6`. No development headers are required.
- RadFoam checks that it was compiled against the running PyTorch version; rebuild it (`pip install .`) after upgrading torch.
- If an import fails after an interrupted build, delete the affected directory under `$TORCH_EXTENSIONS_DIR` (or a stale `lock` file inside it) and try again.

Quick check that everything is in place:

```bash
python -c "import torch, tinycudann, nerfacc, nvdiffrast.torch, radfoam; print('ok', torch.__version__)"
```

## Data

Put (or symlink) all datasets under `load/`. None of the datasets are
redistributed here; ADELE trains directly on the public releases listed below.

### DTU (Table 1)

We use the DTU scans preprocessed by [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
(COLMAP pinhole cameras from the ground-truth poses, images stored as RGBA with
the object mask in the alpha channel). Expected layout:

```
load/dtu/scan24/
  images/0000.png ... 0048.png      # 1554x1162 RGBA (alpha = mask)
  sparse/0/{cameras,images,points3D}.bin
load/dtu_eval/                      # official DTU evaluation data
  ObsMask/ObsMask24_10.mat, Plane24.mat, ...
  Points/stl/stl024_total.ply, ...
```

The 15 evaluation scans are 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110,
114, 118, 122. The official evaluation data (`ObsMask`, `Points/stl`) comes from
the [DTU website](https://roboimagedata.compute.dtu.dk/?page_id=36).

### BlendedMVS (Table 3)

We use the 18 low-resolution object scenes in the IDR layout released with
[Gaussian Surfels](https://github.com/turandai/gaussian_surfels), which also
ships the fused ground-truth point cloud `gt_pts.ply` per scene:

```
load/bmvs/bear/
  image/000.png ...                 # 768x576
  mask/000.png ...
  cameras.npz                       # world_mat_i / scale_mat_i
  gt_pts.ply
```

Scenes: basketball bear bread camera clock cow dog doll dragon durian fountain
gundam house jade man monster sculpture stone.

### NeRF-Synthetic (Table 2)

The standard [NeRF-Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
release, unchanged: `load/nerf_synthetic/<scene>/{train,val,test}/` plus
`transforms_*.json` (800x800 RGBA; the alpha channel is used as the mask).

### Stanford-ORB (supplementary Tables 6 and 7)

The official [Stanford-ORB](https://stanfordorb.github.io/) release:
`load/orb/blender_LDR/<scene>/` (images, masks, `transforms_{train,test}.json`)
and `load/orb/ground_truth/<scene>/mesh_blender/mesh.obj`. The five scenes used in
the paper are `ball_scene002 cactus_scene007 gnome_scene007 pitcher_scene001
teapot_scene006`.

## Training

All paper runs share one configuration; the files under `configs/adele/` differ
only in the dataset block and the scene scale (`model.radius`). See
[configs/README.md](configs/README.md) for the full table.

```bash
# DTU
python launch.py --config configs/adele/dtu.yaml --gpu 0 --train dataset.scene=scan24 trial_name=adele

# BlendedMVS
python launch.py --config configs/adele/bmvs.yaml --gpu 0 --train dataset.scene=bear trial_name=adele

# NeRF-Synthetic (scene-specific radius: chair 1.1, drums 1.2, ficus 1.2, hotdog 1.5,
#                 lego 1.2, materials 1.2, mic 1.3, ship 1.5)
python launch.py --config configs/adele/nerf_synthetic.yaml --gpu 0 --train dataset.scene=lego model.radius=1.2 trial_name=adele

# Stanford-ORB
python launch.py --config configs/adele/orb.yaml --gpu 0 --train dataset.scene=gnome_scene007 trial_name=adele
```

Any config value can be overridden on the command line (`key=value`). Outputs go
to `exp/radtets-<scene>/<trial_name>/`: the extracted mesh is
`save/final_mesh.ply`, checkpoints are under `ckpt/`, validation renders under
`save/`, and TensorBoard logs under `runs/`. A full run takes 4,500 iterations,
about 30 minutes on an H200 and roughly twice that on an RTX 4090.

To fit in **24 GB** use `configs/adele/dtu_24gb.yaml` (one image per batch with
4-step gradient accumulation instead of four images per batch; see
[Memory and runtime](#memory-and-runtime)). The same two overrides work for every
dataset config.

## Evaluation

### DTU

The paper numbers are computed on the raw output mesh after removing geometry
that is not visible from at least two training views (the ground truth only
covers the front-facing surface; see the paper's supplementary material), using
the standard DTU Chamfer protocol:

```bash
python scripts/eval_dtu_all.py --trial adele --gpu 0          # all 15 scans
```

This runs `scripts/filter_dtu_meshes.py` (vertex-visibility filter, `n_min=2`)
and `scripts/eval_dtu/evaluate_single_scene.py` for every scan and writes
`exp/dtu_adele_eval.csv`.

### BlendedMVS

```bash
python scripts/eval_bmvs.py --method_name adele --exp_root ./exp --dataset_path ./load/bmvs
```

### NeRF-Synthetic (extracted-mesh novel-view synthesis)

Bake a per-vertex diffuse albedo from the trained appearance model (32
hemisphere samples), then render the mesh from the test views:

```bash
python scripts/bake_vertex_colors.py --run_dir exp/radtets-lego/adele \
    --mesh exp/radtets-lego/adele/save/final_mesh.ply \
    --out  exp/radtets-lego/adele/save/final_mesh_baked_diffuse32.ply --diffuse_samples 32
python scripts/nerf_synth_nvs.py --trial adele --gpu 0
```

### Stanford-ORB

```bash
python scripts/eval_orb.py --trial adele                       # Chamfer (x2000), culled to training-view visibility
python scripts/bake_vertex_colors.py ... --diffuse_samples 32  # as above, per scene
python scripts/orb_nvs.py --trial adele --gpu 0                 # extracted-mesh PSNR/SSIM
```

## Memory and runtime

Peak GPU memory of the full configuration is about 32 GB on DTU. The
`dtu_24gb.yaml` variant (batch of one image, gradient accumulation over four
steps) peaks below 23 GB and trains faster, at a small cost in accuracy (paper
supplementary, Table 5: 0.53 mm vs 0.56 mm mean Chamfer on DTU). Most of the
footprint comes from the GPU Delaunay triangulation.

Notes:

- Set `--gpu` to the physical GPU index; the launcher sets `CUDA_VISIBLE_DEVICES` itself.
- The first run compiles the JIT extensions (nerfacc, nvdiffrast, renderutils); this takes a few minutes and is cached under `~/.cache/torch_extensions`.
- If a run hangs at "Sanity Checking" for more than a few minutes, a stale lock from an interrupted build is usually the cause: delete `~/.cache/torch_extensions/*/lock` and restart.

## Repository layout

```
launch.py            entry point (train / test / validate)
configs/adele/       paper configurations (one per dataset)
systems/radtets.py   training system: losses, hybrid schedule, densification hooks
models/grid.py       adaptive tetrahedral grid (RadFoam Delaunay, densify / prune)
models/radtets.py    mesh extraction (marching tets), rasterization, depth-offset sampling
models/hashgrid.py, sdf.py, appearance.py   hash-grid SDF and appearance networks
datasets/            COLMAP (DTU), IDR/BlendedMVS, Blender (NeRF-Synthetic, ORB) loaders
render/              nvdiffrast-based rasterizer and the renderutils CUDA plugin (from nvdiffrec)
geometry/dmtet.py    marching tetrahedra (from nvdiffrec)
scripts/             mesh filtering, evaluation, vertex-color baking
```

The code base started from [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)
and keeps its structure: `systems` own the training loop and losses, `models` the
representations, `datasets` the loaders, with OmegaConf configs.

## License

Our code is released under the MIT license (see [LICENSE](LICENSE)). The
repository also contains third-party components under their own licenses, listed
with full texts in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Two of
them are on the training path and restrict use to non-commercial research:
the nvdiffrec renderer (`render/`, `geometry/dmtet.py`; NVIDIA Source Code
License) and the SSIM/NCC helpers in `systems/loss_utils.py` (Inria/MPII
Gaussian-Splatting License). **The software as a whole may therefore only be
used for non-commercial research and evaluation purposes.**

## Citation

```bibtex
@inproceedings{weidenfeller2026adele,
  title     = {{ADELE}: Adaptive Delaunay Grids for High-Fidelity Mesh-Native Reconstruction},
  author    = {Weidenfeller, Johannes and Wang, Shaofei and F{\"u}rnstahl, Philipp and Tang, Siyu},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026}
}
```

## Acknowledgements

This implementation builds on [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl),
[nvdiffrec](https://github.com/NVlabs/nvdiffrec), [RadFoam](https://github.com/theialab/radfoam),
[nvdiffrast](https://github.com/NVlabs/nvdiffrast), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
and [nerfacc](https://github.com/KAIR-BAIR/nerfacc).
