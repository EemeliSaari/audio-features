import inspect

import pytest
import numpy as np

from audio_pipe import transformers
from audio_pipe.factory import load, components


@pytest.mark.parametrize('name', components._available)
def test_invalid_input(name, invalid_data):
    transformer = load(name)

    X = invalid_data.reshape(1, -1)

    with pytest.raises(ValueError):
        if inspect.isclass(transformer):
            # Assume that self is returned after fit
            clf = transformer().fit(X)
            _ = clf.transform(X)
        else:
            _ = transformer(X)


@pytest.mark.parametrize('name', ['minmax', 'normalize'])
def test_valid_input(name, test_mel):
    transformer = load(name)

    X = test_mel.reshape(1, -1)

    if inspect.isclass(transformer):
        # Assume that self is returned after fit
        clf = transformer().fit(X)
        res = clf.transform(X)
    else:
        res = transformer(X)

    assert isinstance(res, np.ndarray)
    assert res.shape == X.shape


@pytest.mark.parametrize('name', ['mel', 'mfcc'])
def test_valid_input(name, test_signal):
    transformer = load(name)

    X = test_signal.reshape(1, -1)

    if inspect.isclass(transformer):
        # Assume that self is returned after fit
        clf = transformer().fit(X)
        res = clf.transform(X)
    else:
        res = transformer(X)

    assert isinstance(res, np.ndarray)
