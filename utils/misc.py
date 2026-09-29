import os
from omegaconf import OmegaConf
from packaging import version


# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('div', lambda a, b: a / b)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
# ======================================================= #


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    print(f"DEBUG: Entering dump_config with path: {path}", flush=True)
    with open(path, 'w') as fp:
        print(f"DEBUG: Starting OmegaConf.save to {path}", flush=True)
        OmegaConf.save(config=config, f=fp)
        print(f"DEBUG: Finished OmegaConf.save to {path}", flush=True)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def parse_version(ver):
    return version.parse(ver)



def C(value, global_step, current_epoch):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        elif isinstance(end_step, float):
            current_step = current_epoch
            value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
    return value