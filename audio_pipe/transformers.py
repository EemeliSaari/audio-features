from dataclasses import dataclass

import numpy as np
from librosa.feature import melspectrogram, mfcc
from librosa.util import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array as check_2d
from sklearn.utils.validation import check_is_fitted

from .factory import register
from .pipeline import Component


def check_array(X: np.ndarray):
    """check_array
    
    Similar function to `sklearn.utils.check_array` but for 3d input.

    Asserts that a given array is a numeric tensor of shape:
        - (feature, feature, observation).

    Parameters
    ----------
    X: np.ndarray
        Input array to validate
    """
    if len(X.shape) < 3:
        raise ValueError('Expecting 3d array input')
    for i in range(X.shape[0]):
        check_2d(X[i, :, :])


@dataclass
@register(name='normalize')
class Normalizer(TransformerMixin, BaseEstimator, Component):
    """Normalizer
    
    A simple normalizer for audio spectograms wrapping the utility
    function `librosa.utils.normalize`.

    Parameters
    ----------
    [Reference](https://librosa.org/doc/main/generated/librosa.util.normalize.html)
    """
    axis: int = 0
    norm: float = np.inf
    threshold: float = None
    fill: bool = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_array(X)

        out = np.zeros(X.shape, dtype=X.dtype)
        for i in range(X.shape[0]):
            out[i, :, :] = normalize(X[i, :, :], norm=self.norm,
                                     axis=self.axis, fill=self.fill,
                                     threshold=self.threshold)
        return out


@dataclass
@register(name='minmax')
class MinMaxScaler(TransformerMixin, BaseEstimator, Component):
    """MinMaxScaler

    Scales the values between given target min and max values from the given
    set of observations.

    Parameters
    ----------
    target_min: float, default=0.0
        Target minimum value
    target_max: float, default=1.0
        Target maximum value
    """
    target_min: float = 0.0
    target_max: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> TransformerMixin:
        check_array(X)
        self.max_ = X.max()
        self.min_ = X.min()
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_array(X)
        check_is_fitted(self)
        std = (X - self.target_min) / (self.target_max - self.target_min)
        return std * (self.max_ - self.min_) + self.min_


@dataclass
@register(name='mel')
class MELTransformer(TransformerMixin, BaseEstimator, Component):
    """MELTransformer

    Wrapper for `librosa` mel-scaled spectogram based.

    Parameters
    ----------
    [Reference](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram)
    """
    hop_length: int = 512
    center: bool = True
    pad_mode: str = 'reflect'

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> TransformerMixin:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_2d(X)

        out = []
        for i in range(X.shape[0]):
            trs = melspectrogram(y=X[i, :], hop_length=self.hop_length,
                                 center=self.center, pad_mode=self.pad_mode)
            out.append(trs.tolist())
        return np.array(out)


@dataclass
@register(name='mfcc')
class MFCCTransformer(TransformerMixin, BaseEstimator, Component):
    """MFCCTransformer

    Wrapper for `librosa`  mel-frequency cepstral coefficients

    Parameters
    ----------
    [Reference](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)
    """
    sr: int = 22050
    n_mfcc: int = 20
    hop_length: int = 512
    win_length: int = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> TransformerMixin:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_2d(X)

        out = []
        for i in range(X.shape[0]):
            trs = mfcc(y=X[i, :], sr=self.sr, n_mfcc=self.n_mfcc,
                        hop_length=self.hop_length, win_length=self.win_length)
            out.append(trs.tolist())

        return np.array(out)
