import os
import subprocess
import shutil
from utils.misc import dump_config, parse_version


import pytorch_lightning
if parse_version(pytorch_lightning.__version__) > parse_version('1.8'):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch


class MemoryLoggingCallback(Callback):
    def __init__(self, snapshot_step=None):
        super().__init__()
        self.snapshot_step = snapshot_step
        self.history_started = False

    def _get_cpu_memory(self):
        # Fallback for systems without psutil
        try:
            with open('/proc/self/status', 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0  # Convert KB to MB
        except:
            pass
        return 0.0

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start recording history 10 steps before the target snapshot step
        if self.snapshot_step is not None and not self.history_started:
            if trainer.global_step >= self.snapshot_step - 10:
                print(f"Starting memory history recording (step {trainer.global_step})...")
                torch.cuda.memory._record_memory_history(max_entries=100000)
                self.history_started = True

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.snapshot_step is not None and self.history_started:
            if trainer.global_step == self.snapshot_step:
                snapshot_file = os.path.join(trainer.default_root_dir, f"memory_snapshot_step_{self.snapshot_step}.pkl")
                print(f"Dumping memory snapshot to {snapshot_file}...")
                torch.cuda.memory._dump_snapshot(snapshot_file)
                # Stop recording to save resources
                torch.cuda.memory._record_memory_history(enabled=None)

        if trainer.global_step % trainer.log_every_n_steps == 0:
            gpu_stats = torch.cuda.memory_stats()
            
            gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            gpu_mem_max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            # Gap analysis metrics
            # Inactive memory is memory that is reserved but not currently allocated
            inactive_mem = (gpu_mem_reserved - gpu_mem_alloc)
            
            # Non-releasable memory (fragmentation)
            inactive_split_mb = gpu_stats.get('inactive_split.all.current', 0) / (1024 ** 2)
            
            cpu_mem = self._get_cpu_memory()

            pl_module.log('memory/gpu_allocated_mb', gpu_mem_alloc, on_step=True, on_epoch=False, prog_bar=False)
            pl_module.log('memory/gpu_reserved_mb', gpu_mem_reserved, on_step=True, on_epoch=False, prog_bar=False)
            pl_module.log('memory/gpu_max_allocated_mb', gpu_mem_max_alloc, on_step=True, on_epoch=False, prog_bar=False)
            pl_module.log('memory/gpu_inactive_mb', inactive_mem, on_step=True, on_epoch=False, prog_bar=False)
            pl_module.log('memory/gpu_inactive_split_mb', inactive_split_mb, on_step=True, on_epoch=False, prog_bar=False)
            pl_module.log('memory/cpu_res_mb', cpu_mem, on_step=True, on_epoch=False, prog_bar=False)

            # Reset peak memory stats periodically to track current peak
            if trainer.global_step % (trainer.log_every_n_steps * 10) == 0:
                torch.cuda.reset_peak_memory_stats()


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1
    
    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(self.save_root, self.version if isinstance(self.version, str) else f"version_{self.version}")


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
    
    def get_file_list(self):
        return [
            b.decode() for b in
            set(subprocess.check_output('git ls-files', shell=True).splitlines()) |
            set(subprocess.check_output('git ls-files --others --exclude-standard', shell=True).splitlines())
        ]
    
    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn("Code snapshot is not saved. Please make sure you have git installed and are in a git repository.")


class ConfigSnapshotCallback(VersionedCallback):
    def __init__(self, config, save_root, version=None, use_version=True):
        super().__init__(save_root, version, use_version)
        self.config = config

    @rank_zero_only
    def save_config_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        dump_config(os.path.join(self.savedir, 'parsed.yaml'), self.config)
        shutil.copyfile(self.config.cmd_args['config'], os.path.join(self.savedir, 'raw.yaml'))


    def on_fit_start(self, trainer, pl_module):
        self.save_config_snapshot()


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items
