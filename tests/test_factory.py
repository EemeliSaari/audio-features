import pytest

from audio_pipe import transformers
from audio_pipe.factory import load, components


def test_component_not_found():
    with pytest.raises(ValueError):
        _ = load('missing')


@pytest.mark.parametrize('name', components._available)
def test_valid_load(name):
    _ = load(name)


def test_multiple_imports():
    import audio_pipe.transformers as transformers
    import audio_pipe.transformers as transformers
