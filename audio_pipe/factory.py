from typing import Any, Dict, Callable


def load(name: str) -> Any:
    """load

    Load the component class/function with a given name.
    """
    if name not in components._available:
        raise ValueError(f'Component \'{name}\' not found')
    return components._available[name]


def register(cls: Callable = None, /, name: str = None):
    """register

    Registers a pipeline component for the singleton

    Parameters
    ----------
    cls: Callable
        Function or a class
    name: str, default=None
        Name given defaulting to function/class name
    """
    if not name:
        name = cls.__name__

    def wrap(cls):
        if name in components._available:
            raise ValueError(f'Module {name} already registered')
        components._available[name] = cls
        return cls

    if cls is None:
        return wrap

    return wrap(cls)


class components:
    """components

    A singleton class that allows for registering
    different components as part of factory.
    """
    _available: Dict[str, Any] = {}
