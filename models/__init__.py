models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model


from . import radtets, hashgrid, grid, appearance, environment_map, sdf, modulator, background_sphere
