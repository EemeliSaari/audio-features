from configparser import ConfigParser

import pytest

from audio_pipe.pipeline import Pipeline, load_config


def test_valid_config():
    conf = ConfigParser()
    conf.read_string("""
        [pipe]
        steps = ["first", "second"]
        type = pipeline

        [first]
        name = mel
        type = component

        [second]
        name = minmax
        type = component
    """)

    res = load_config(conf)
    assert all(isinstance(x, Pipeline) for x in res)


def test_missing_type():
    conf = ConfigParser()
    conf.read_string("""
        [pipe]
        steps = ["first", "second"]
        type = pipeline

        [first]
        name = mel

        [second]
        name = minmax
        type = component
    """)
    with pytest.raises(ValueError):
        _ = load_config(conf)


def test_invalid_type():
    conf = ConfigParser()
    conf.read_string("""
        [pipe]
        steps = ["first", "second"]
        type = pipeline

        [first]
        name = mel
        type = unknown

        [second]
        name = minmax
        type = component
    """)
    with pytest.raises(ValueError):
        _ = load_config(conf)
