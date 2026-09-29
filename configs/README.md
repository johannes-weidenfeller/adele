# Configs

All configs are OmegaConf YAML for `launch.py`:

```
python launch.py --config <config.yaml> --gpu <id> --train [key=value ...]
```

Outputs go to `./exp/<name>/<trial_name>/` with `name = radtets-<dataset.scene>`; pass
`trial_name=<something>` to give the run a fixed folder (default: `tag@timestamp`). Every
config takes `dataset.scene=<scene>` on the command line (`???` in the file); the per-scene
overrides beyond that are listed in the tables below.

The method block (everything under `model:` and `system:` plus checkpoint/export/trainer)
is identical across all configs; only the `dataset:` block and the scene scale
`model.radius` differ between datasets.

## ADELE (`configs/adele/`)

| Config | Paper table | Per-scene CLI override | Data expected under `dataset.root_dir` |
|---|---|---|---|
| `dtu.yaml` | Table 1 (DTU, 49 views) | `dataset.scene=scan24` | `./load/dtu/<scan>/`: COLMAP `sparse/0/`, `images/*.png` (RGBA; mask = alpha), 1554x1162, trained at `img_downscale: 2` |
| `dtu_24gb.yaml` | Table 1, memory-efficient variant | `dataset.scene=scan24` | as `dtu.yaml` |
| `bmvs.yaml` | Table 3 (BlendedMVS) | `dataset.scene=bear` | `./load/bmvs/<scene>/`: `cameras.npz`, `image/{i:03d}.png` (768x576), `mask/` |
| `nerf_synthetic.yaml` | Table 2 (NeRF-synthetic) | `dataset.scene=lego model.radius=1.2` (radius per scene in `scenes_nerf_synthetic.yaml`: chair 1.1, drums 1.2, ficus 1.2, hotdog 1.5, lego 1.2, materials 1.2, mic 1.3, ship 1.5) | `./load/nerf_synthetic/<scene>/`: `transforms_{train,val,test}.json`, RGBA PNGs (mask = alpha) |
| `orb.yaml` | Supp. Tables 6/7 (Stanford-ORB) | `dataset.scene=<one of>` `ball_scene002 cactus_scene007 gnome_scene007 pitcher_scene001 teapot_scene006` | `./load/orb/blender_LDR/<scene>/`: `transforms_*.json`, `train/`, `train_mask/` |

Batch size: `model.train_num_images` images per micro-batch times
`trainer.accumulate_grad_batches` micro-batches per optimizer step. `dtu.yaml` uses 4 x 4;
`dtu_24gb.yaml` is the same config with `model.train_num_images: 1` (1 image per
micro-batch, gradient accumulation over 4 micro-batches), which fits a 24 GB GPU.

Examples:

```
python launch.py --config configs/adele/dtu.yaml            --gpu 0 --train dataset.scene=scan24
python launch.py --config configs/adele/dtu_24gb.yaml       --gpu 0 --train dataset.scene=scan24
python launch.py --config configs/adele/bmvs.yaml           --gpu 0 --train dataset.scene=bear
python launch.py --config configs/adele/nerf_synthetic.yaml --gpu 0 --train dataset.scene=lego model.radius=1.2
python launch.py --config configs/adele/orb.yaml            --gpu 0 --train dataset.scene=gnome_scene007
```
