
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-10-10
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
_model_entrypoints = {}
_backbone_entrypoints = {}
_unavailable_backbones = {}


def register_model(fn):
    model_name = fn.__name__
    _model_entrypoints[model_name] = fn
    return fn


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def create_model(model_name, **kwargs):
    create_fn = _model_entrypoints[model_name]
    model = create_fn(**kwargs)
    return model


def register_backbone(feat_dim):
    def wrapper(fn):
        backbone_name = fn.__name__
        _backbone_entrypoints[backbone_name] = (fn, feat_dim)
        _unavailable_backbones.pop(backbone_name, None)
        return fn
    return wrapper


def register_unavailable_backbones(backbone_names, dependency, error):
    """Record optional backbones whose import dependency is unavailable."""
    for backbone_name in backbone_names:
        _unavailable_backbones[backbone_name] = (dependency, error)


def backbone_entrypoint(bb_name):
    if bb_name in _unavailable_backbones:
        dependency, error = _unavailable_backbones[bb_name]
        raise RuntimeError(
            f"Backbone '{bb_name}' requires optional dependency '{dependency}'"
        ) from error
    return _backbone_entrypoints[bb_name]


def create_backbone(bb_name, **kwargs):
    create_fn, feat_dim = backbone_entrypoint(bb_name)
    backbone = create_fn(True, **kwargs)
    return backbone, feat_dim
