import sys
import argparse
import os
import time
import logging
from datetime import datetime

# PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the
# omegaconf config objects embedded in our Lightning checkpoints on --resume.
# These are our own trusted local checkpoints, so restore the pre-2.6 behaviour.
import torch as _torch
if not getattr(_torch.load, '_wo_shim', False):
    _orig_load = _torch.load
    def _patched_load(*a, **k):
        k.setdefault('weights_only', False)
        return _orig_load(*a, **k)
    _patched_load._wo_shim = True
    _torch.load = _patched_load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--predict', action='store_true')
    # group.add_argument('--export', action='store_true') # TODO: a separate export action

    parser.add_argument('--exp_dir', default='./exp')
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')
    parser.add_argument('--profile', action='store_true', help='if true, enable PyTorchProfiler')
    parser.add_argument('--snapshot_step', type=int, default=None, help='step at which to take a memory snapshot')

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    # Preload the renderutils CUDA plugin here, BEFORE importing systems (which
    # pulls in tinycudann and reserves a large virtual address space). The plugin
    # otherwise loads lazily at the first mesh render; torch's `ninja --version`
    # check forks the process, and under this machine's memory-overcommit limit
    # that fork is rejected once the process is large (esp. high-res datasets like
    # mip360) and misreported as "Ninja is required". With only torch imported the
    # fork is cheap; the loaded plugin is cached for the rest of the run.
    import torch  # noqa: F401
    try:
        from render.renderutils import ops as _ru_ops
        _ru_ops._get_plugin()
    except Exception as _e:
        print(f"[preload] renderutils plugin preload skipped ({_e}); will load lazily")

    import datasets
    import systems
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.profiler import PyTorchProfiler, SimpleProfiler
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar, MemoryLoggingCallback
    from utils.misc import load_config

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    # if not config.dataset.get('normalize_cameras', True):
    #     scale = dm.train_dataset.scale 
    #     transform = dm.train_dataset.transform
    #     config.model.scale = scale*0.6
    #     config.model.offset = torch.linalg.inv(transform)[:3,3].cpu().numpy()
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)
    #system.dataset = dm.train_dataloader().dataset

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            LearningRateMonitor(logging_interval='step'),
            # CodeSnapshotCallback(
            #     config.code_dir, use_version=False
            # ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=20),
            MemoryLoggingCallback(snapshot_step=args.snapshot_step),
        ]

    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(args.runs_dir, name=config.name, version=config.trial_name),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
        ]
    
    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        #strategy = 'ddp_find_unused_parameters_false'
        #strategy = 'ddp'
        strategy = 'auto'
        #strategy = None
        #strategy = 'dp' #Debugging


    # Set up profiler
    if args.profile or hasattr(config, 'profiler'):
        if hasattr(config, 'profiler'):
            profiler_cfg = dict(config.profiler)
        else:
            # Default profiler config if enabled via CLI but not in config
            profiler_cfg = {
                'filename': 'profiler_trace',
                'export_to_chrome': True,
                'profile_memory': True,
                'schedule': torch.profiler.schedule(wait=50, warmup=5, active=5, repeat=1)
            }

        # Optional: translate a YAML schedule dict into torch.profiler.schedule(...)
        schedule_cfg = profiler_cfg.pop("schedule", None)

        if schedule_cfg is not None:
            import torch
            profiler_cfg["schedule"] = torch.profiler.schedule(**schedule_cfg)
        profiler = PyTorchProfiler(**profiler_cfg)
    else:
        profiler = None
    
    trainer = Trainer(
        profiler = profiler,
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        **config.trainer
    )

    if args.train:
        if args.resume and not args.resume_weights_only:
            # FIXME: different behavior in pytorch-lighting>1.9 ?
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
