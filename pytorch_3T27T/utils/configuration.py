#!/usr/bin/env python
# coding=utf-8

from os.path import join as pjoin
from typing import Union, List, Any, Sequence
from types import ModuleType
from numbers import Number
import inspect
from collections import namedtuple
import re
import yaml
from .backer import get_etc_path, get_trial_dict

__all__ = [
    'Configuration', 'load_configuration_from_yaml',
    'save_configuration_to_yaml', 'init_and_config', 'configure',
    'introspect_constructor'
]

_initialize_t = Union[List[Any], Any]


class Configuration:
    """Base class for all configurations"""

    def __init__(self, *args, **options):
        self.args = []
        self.options = {}
        self.set(*args, **options)

    def get(self, key, default_value=None):
        """Get option"""
        return self.options.get(key, default_value)

    def set(self, *args, **options):
        """Set arguments and options"""

        self.set_args(*args)
        self.set_options(**options)
        return self

    def filter_args(self, *args):
        """Keep only the arguments provided"""
        to_rm_args = [arg for arg in self.args if arg not in args]
        self.remove_args(*to_rm_args)

        return self

    def filter_options(self, *keys):
        """Keep only the options provided"""
        to_rm_opts = [opt for opt in self.options if opt not in keys]
        self.remove_options(*to_rm_opts)

        return self

    def set_args(self, *args):
        """Set arguments"""

        for a in args:
            self.args.append(a)
        return self

    def set_options(self, **options):
        """Set options"""

        for k, v in options.items():
            self.options[k] = v
        return self

    def remove(self, *params):
        """Remove arguments and options"""

        self.remove_args(*params)
        self.remove_options(*params)
        return self

    def remove_args(self, *args):
        """Remove arguments"""

        for arg in args:
            if arg in self.args:
                self.args.remove(arg)
        return self

    def remove_options(self, *keys):
        """Remove options"""

        for key in keys:
            if key in self.options:
                del self.options[key]
        return self

    def update(self, *args, **options):
        """Update arguments and options"""

        self.update_args(*args)
        self.update_options(**options)
        return self

    def update_args(self, *args):
        """Update arguments and options"""

        for arg in args:
            if arg not in self.args:
                self.args += [arg]
        return self

    def update_options(self, **options):
        """Update options"""

        self.options.update(options)
        return self

    def __str__(self):
        args = []
        for arg in self.args:
            if not isinstance(arg, (Number, str, Sequence)):
                args += [retrieve_name(arg)]
            else:
                args += [arg]
        string = " ".join(args)
        for k, v in self.options.items():
            if not isinstance(v, (Number, str, Sequence)):
                v = retrieve_name(v)
            string += f" --{k} {v}"
        return string

    def clone(self):
        """
        Clone configuration

        Returns
        -------
        cfg : Configuration
        """

        cfg = Configuration()
        cfg.args = self.args.copy()
        cfg.options = self.options.copy()
        return cfg


def load_configuration_from_yaml(config_file: str):
    """
    Load configuration from YAML file and update trial information

    Parameters
    ----------
    config_file : str
        Path to yaml file with configurations.

    Returns
    -------
    cfg : Configuration
        Configuration object with parameters and options set according to yaml
        configuration file.
    """

    cfg = Configuration()
    with open(config_file) as handle:
        config_from_file = yaml.safe_load(handle)

    cfg.options = config_from_file
    if 'args' in cfg.options:
        cfg.set_args(*cfg.options['args'])
        cfg.remove_options('args')

    # Get trial information from file name
    trial = get_trial_dict(config_file)
    cfg.set_options(trial=trial)

    return cfg


def save_configuration_to_yaml(cfg: Configuration, save_dir: str = None):
    """
    Save configuration options to YAML file

    Parameters
    ----------
    cfg : Configuration
    save_dir : str
        Path to directory where YAML configuration file will be saved
    """

    # if save_dir is None, save configuration file into etc directory:
    if save_dir is None:
        save_dir = get_etc_path(cfg.config)
    cfg_filename = cfg.options["trial"]["ID"] + ".yml"
    cfg_path = pjoin(save_dir, cfg_filename)
    with open(cfg_path, 'w') as handle:
        yaml_object = {'args': cfg.args, **cfg.options}
        yaml.dump(yaml_object, handle, default_flow_style=False)


def init_and_config(module: ModuleType, constructor_type: str,
                    cfg: Configuration, *args, **kwargs) -> Any:

    available_types = ['transforms', 'dataset', 'dataloader', 'network',
                       'loss', 'lr_scheduler', 'optimizer', 'metrics']

    if get_generic_type(constructor_type) not in available_types:
        raise KeyError(f'constructor type must be one of {available_types}')

    if constructor_type not in cfg.options:
        msg = 'Constructor type missing from provided configuration file'
        raise AttributeError(msg)

    if not isinstance(cfg.options[constructor_type], list):
        recipes = [cfg.options[constructor_type]]
    else:
        recipes = cfg.options[constructor_type]

    cfg_list = [Configuration(**recipe) for recipe in recipes]

    instances = [initialize(module, constructor_type, cfg_, *args, **kwargs)
                 for cfg_ in cfg_list]

    configured_instances = [configure(instance, cfg=cfg_, *args, **kwargs)
                            for (instance, cfg_) in zip(instances, cfg_list)]

    if len(configured_instances) == 1:
        return configured_instances[0]


# TODO find module and args and kwargs of init method by introspection
def initialize(module: ModuleType, constructor_type: str,
               cfg: Configuration, *args, **kwargs) -> _initialize_t:
    """
    Helper to construct an instance of a class from Configuration object

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    constructor_type : str
        A key of cfg.options. One of 'transforms', 'dataset', 'dataloader',
        'network', 'loss', 'lr_scheduler', 'optimizer', 'metrics'
    cfg : Configuration
        Object with the positional arguments (cfg.args) and keyword arguments
        (cfg.opts) used to construct the class instance.
    args : list
        Runtime positional arguments used to construct the class instance.
    kwargs : dict
        Runtime keyword arguments used to construct the class instance.

    Returns
    -------
    _ : Any
        Instance of module.
    """

    constructor_name = cfg.get('type', None)
    cfg_args = cfg.get('args', [])
    cfg_opts = cfg.get('options', {})
    cfg_ = Configuration(*args, **kwargs)
    cfg_.update(*cfg_args, **cfg_opts)


    argspec = introspect_constructor(constructor_name, module)
    #TODO figure out how to filter arguments properly
    # if argspec.varargs is None:
    #     # Then the class does not support variable arguments
    #     cfg_.filter_args(*argspec.args)
    if argspec.keywords is None:
        # Then the class does not support variable keywords
        cfg_.filter_options(*argspec.defaults.keys())

    return get_instance(module, constructor_name, *cfg_.args, **cfg_.options)


def introspect_constructor(class_name: Any, module_name: str = None):
    if module_name:
        func = getattr(module_name, class_name, '__init__')
    else:
        func = getattr(class_name, '__init__')
    sig = inspect.signature(func.__init__)
    defaults = {
        p.name: p.default for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not p.empty
    } or None
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
        p.name != 'self'
    ]
    # Only keep the non default parameters
    args = list(filter(lambda arg: arg not in defaults, args))

    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = keywords[0] if keywords else None

    argspec = namedtuple('Signature', ['args', 'defaults',
                                       'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords)


def get_instance(module: ModuleType, constructor_name: str,
                 *args, **kwargs) -> Any:
    """
    Helper to construct an instance of a class.

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    constructor_name : str
        Name of class, as would be returned by ``.__class__.__name__``.
    args : list
        Positional arguments used to construct the class instance.
    kwargs : dict
        Keyword arguments used to construct the class instance.
    """
    return getattr(module, constructor_name)(*args, **kwargs)


def configure(obj: Any, *args, **options) -> Any:

    # Create object Configuration attribute with object default params, if any
    if not hasattr(obj, 'cfg'):
        if not hasattr(obj, 'default_params'):
            obj.cfg = Configuration()
        else:
            default_args, default_options = obj.default_params()
            obj.cfg = Configuration(*default_args, **default_options)

    cfg = options.pop('cfg', None)

    # Update params with supplied params, if any
    obj.cfg.update(*args, **options)

    # Update params with supplied Configuration, if any
    if cfg and isinstance(cfg, Configuration):
        obj.cfg.update(*cfg.get('args', []), **cfg.get('options', {}))

    if obj.__class__.__name__ == 'AugmentationFactory':
        obj.cfg.set(*obj.cfg.options['augmentations'])
        obj.cfg.remove_options('augmentations', 'train')

    return obj


def get_generic_type(given_name: str) -> str:
    obj_types = ['transforms', 'dataset', 'dataloader', 'network', 'loss',
                 'lr_scheduler', 'optimizer', 'metrics']
    pattern = '(' + "|".join(obj_types) + ')'
    regex = re.compile(pattern)
    obj_type = regex.search(given_name)
    if obj_type:
        # If there is a match, its in a sigleton tupple
        obj_type = obj_type.groups()[0]
    return obj_type


def retrieve_name(var: Any) -> str:
    """
    Gets the name of var. Does it from the out most frame inner-wards.

    Parameters
    ----------
    var: Any
        Variable to get name from.

    Returns
    -------
    _ : str
        Variable given name
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items()
                 if var_val is var]
        if len(names) > 0:
            return names[0]
