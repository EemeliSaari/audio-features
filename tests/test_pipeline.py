import numpy as np
import pytest
from audio_pipe import (MELTransformer, MFCCTransformer, MinMaxScaler,
                        Normalizer, pipeline)


def test_mfcc_pipeline_minmax(test_signal):
    pipe = pipeline.Pipeline([MFCCTransformer(), MinMaxScaler()])
    X = test_signal.reshape(1, -1)
    res = pipe(X)
    assert isinstance(res, np.ndarray)


def test_mfcc_pipeline_stand(test_signal):
    pipe = pipeline.Pipeline([MFCCTransformer(), Normalizer()])
    X = test_signal.reshape(1, -1)
    res = pipe(X)
    assert isinstance(res, np.ndarray)


def test_mel_pipeline_minmax(test_signal):
    pipe = pipeline.Pipeline([MELTransformer(), MinMaxScaler()])
    X = test_signal.reshape(1, -1)
    res = pipe(X)
    assert isinstance(res, np.ndarray)


def test_mel_pipeline_stand(test_signal):
    pipe = pipeline.Pipeline([MELTransformer(), Normalizer()])
    X = test_signal.reshape(1, -1)
    res = pipe(X)
    assert isinstance(res, np.ndarray)
