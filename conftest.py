import pathlib

import pytest
import librosa
import numpy as np


@pytest.fixture
def test_signal(request):
    y, sr = librosa.load(librosa.example('trumpet'))
    yield y


@pytest.fixture
def test_mel(test_signal):
    yield librosa.feature.melspectrogram(test_signal)


@pytest.fixture
def invalid_data(request):
    return np.array([['asd', 2], [0, 3.13]])
