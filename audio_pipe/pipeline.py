import configparser
import json
import pathlib
from configparser import ConfigParser
from dataclasses import dataclass, fields
from typing import List, Union

import numpy as np

from . import factory


@dataclass
class Component:
    """Component

    Abstraction for component to convert type at the runtime
    based on the annotations for dataclass.
    """
    name: str = None

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if not value:
                continue
            setattr(self, f.name, f.type(value))


@dataclass
class Pipeline:
    """Pipeline

    Abstraction to reduce data based on list of callable functions
    defined in the steps.
    """
    steps: List[Component]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X_ = X
        for step in self.steps:
            if hasattr(step, 'fit_transform'):
                X_ = step.fit_transform(X_)
            else:
                X_ = step(X_)
        return X_

    @property
    def name(self):
        return '_'.join(x.name for x in self.steps)


def load(path: Union[pathlib.Path, str]) -> List[Pipeline]:
    """load

    Load a list of Pipelines from file.

    Parameters
    ----------
    path: pathlike
        Path to configuration

    Returns
    -------
    pipelines: list of Pipeline
        Initialized pipelines
    """
    cfg = ConfigParser(strict=True)
    with open(path, 'r') as f:
        cfg.read_file(f)

    pipelines = load_config(cfg)

    return pipelines


def load_config(cfg: ConfigParser) -> List[Pipeline]:
    """load_config
    
    Parses the configuration file for valid pipelines and components.

    Parameters
    ----------
    cfg: configparser.ConfigParser
        Initialized and loaded parser

    Returns
    -------
    pipelines: list of Pipeline
        Initialized pipelines
    """
    pipecfg, components = {}, {}
    for section in cfg.sections():
        stype = cfg[section].pop('type', '')

        params = dict(cfg[section].items())
        if stype == 'pipeline':
            pipecfg[section] = dict(cfg[section].items())
        elif stype == 'component':
            name = params.pop('name', None)
            cls = factory.load(name)
            components[section] = cls(name=section, **params)
        else:
            raise ValueError(f'Invalid section: {section}')

    pipelines = []
    for name, conf in pipecfg.items():
        steps = [components[s] for s in json.loads(conf['steps'])]
        pipelines.append(Pipeline(steps))
    return pipelines
