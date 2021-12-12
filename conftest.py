import pathlib

import pytest
import librosa
import numpy as np


def pytest_addoption(parser):
    parser.addoption('--datapath', action='store', default='genres')


@pytest.fixture
def test_signal(request):
    path = pathlib.Path(request.config.option.datapath)
    test_file = path.joinpath('blues').joinpath('blues.00000.wav')
    y, sr = librosa.load(test_file)
    yield y


@pytest.fixture
def test_mel(test_signal):
    yield librosa.feature.melspectrogram(test_signal)


@pytest.fixture
def invalid_data(request):
    return np.array([['asd', 2], [0, 3.13]])
